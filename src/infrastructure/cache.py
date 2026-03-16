"""
Redis cache layer — cache-aside pattern with TTL-based expiry.

Cache key convention:
  account:{org_id}:{account_id}   → account projection JSON  (TTL=5min)
  integrations:{org_id}           → list of integrations      (TTL=10min)
  member:{org_id}:{user_id}       → org member role           (TTL=2min)

Why cache-aside (lazy population) over write-through?
  - Account balance is updated by Kafka consumer asynchronously.
  - We can't write the cache at write time because the projection isn't built yet.
  - Cache-aside: populate on first read, invalidate when consumer updates projection.
  - This matches our CQRS model: consumer owns the projection + cache invalidation.

Thundering herd: mitigated by short TTLs (5 min) + consumer invalidation.
In production: add probabilistic early expiry (XFetch algorithm) if needed.
"""
import json
import redis.asyncio as aioredis
from typing import Optional, Any


class CacheClient:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if key does not exist / is expired."""
        value = await self._redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Store value as JSON with an absolute TTL (SETEX)."""
        await self._redis.setex(key, ttl_seconds, json.dumps(value, default=str))

    async def delete(self, key: str) -> None:
        """Remove a single key (cache invalidation on mutation)."""
        await self._redis.delete(key)

    async def delete_pattern(self, pattern: str) -> None:
        """
        Invalidate all keys matching a glob pattern.

        Example: delete_pattern("account:org-uuid:*") clears all accounts for one org.
        Uses KEYS — acceptable for low-cardinality invalidation (org-scoped).
        For high-cardinality patterns in production, use SCAN instead.
        """
        keys = await self._redis.keys(pattern)
        if keys:
            await self._redis.delete(*keys)

    async def ping(self) -> bool:
        """Health check — returns True if Redis is reachable."""
        try:
            return await self._redis.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Graceful shutdown — release connection pool."""
        await self._redis.aclose()
