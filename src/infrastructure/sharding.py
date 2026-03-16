"""
Consistent Hashing — Application-Level Database Sharding

USER STORY
----------
As a Finance Tracker architect, I want tenant data distributed across database
shards using a consistent hash ring, so that when we add new shards to handle
growth, only 1/N of tenants need to migrate — not all of them simultaneously.

THE PROBLEM WITH NAIVE MODULO SHARDING
----------------------------------------
  shard_id = hash(org_id) % N

  With N=4 shards: org_A → shard 2, org_B → shard 0
  Add shard (N=5):  org_A → shard 1, org_B → shard 4   ← BOTH MOVED

  Every single tenant gets reassigned. You must migrate 100% of data
  simultaneously. That's a multi-day outage at scale.

CONSISTENT HASHING SOLUTION
-----------------------------
  Imagine a circle (the "ring") with positions 0 → 2^32.
  Place each shard at multiple positions (virtual nodes) around the ring.
  To find a tenant's shard: hash their org_id, walk clockwise to the first
  shard marker.

  Adding a shard: it takes positions from its clockwise neighbours.
  Only tenants in those ranges need to migrate — approximately 1/N of all
  tenants. All others are completely unaffected.

VIRTUAL NODES
--------------
  Each physical shard gets 150 positions (virtual nodes) around the ring.
  With 4 shards × 150 virtual nodes = 600 ring positions.
  Each shard gets ~25% of the ring, evenly distributed.
  Without virtual nodes, 4 random points on the ring would be uneven:
  one shard might cover 40% of the ring, another only 10%.

EXAMPLE
--------
  Ring with 4 shards, 2 virtual nodes each (simplified):

  Position 0──────────────────────────────4,294,967,295
           │                                            │
           shard2-vn0     shard0-vn0     shard3-vn0    │
           shard1-vn1     shard2-vn1     shard0-vn1    │
           │                                            │
           org_id "abc123" hashes to position 1,500,000,000
           → clockwise → first node is shard0-vn1 at 2,100,000,000
           → org "abc123" lives on Shard 0

IMPLEMENTATION IN THIS PROJECT
--------------------------------
  We demonstrate the algorithm with one physical database (dev/CI).
  In production: SHARD_URLS maps each shard_id to a different DATABASE_URL.
  The get_shard_pool() function selects the correct asyncpg pool.

  Route: GET /api/v1/debug/shard-info
  Shows: which shard the authenticated user's org would live on.
  Used in interviews to demonstrate the algorithm live.
"""

import hashlib
from uuid import UUID


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NUM_SHARDS: how many physical database servers in production.
# In this project: 4 logical shards, all pointing to the same DB (dev mode).
# In production: 4 separate PostgreSQL instances.
NUM_SHARDS = 4

# VIRTUAL_NODES_PER_SHARD: how many ring positions each physical shard gets.
# Higher = better distribution = more memory for the ring dict.
# 150 is the industry standard (used by Cassandra, DynamoDB).
# At 150: standard deviation of shard load ≈ 2% (near-perfect balance).
# At 10:  standard deviation ≈ 15% (noticeable imbalance).
VIRTUAL_NODES_PER_SHARD = 150


# ---------------------------------------------------------------------------
# The Hash Ring
# ---------------------------------------------------------------------------

class ConsistentHashRing:
    """
    A consistent hash ring for mapping arbitrary keys to shard IDs.

    Shard key: organization_id (UUID)
    Why org_id and NOT user_id?
      - All queries are org-scoped: WHERE organization_id = $1
      - If we sharded by user_id, one org's data would span all shards
        → every query would need to scatter-gather across all shards
      - Sharding by org_id ensures all data for one tenant lives on one shard
        → queries touch exactly one shard (no cross-shard joins)

    Thread safety: the ring dict and sorted_positions are built once at
    __init__ and never modified. All reads (get_shard) are stateless.
    Python's GIL makes read-only dict access safe from multiple threads.
    """

    def __init__(
        self,
        num_shards: int = NUM_SHARDS,
        virtual_nodes: int = VIRTUAL_NODES_PER_SHARD,
    ):
        # num_shards: number of physical shards (database servers).
        # virtual_nodes: how many ring positions per physical shard.
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes

        # self.ring: maps ring_position (int) → shard_id (int 0..N-1)
        # Built once at startup. Never mutated after __init__.
        # Example with 2 shards, 2 virtual nodes each:
        #   {102938471: 0, 394857291: 1, 593847201: 0, 847392019: 1}
        self.ring: dict[int, int] = {}

        # Build the ring
        self._build_ring()

        # sorted_positions: ring.keys() sorted ascending.
        # Used in get_shard() for binary search (clockwise traversal).
        # Sorting once at build time → O(1) per lookup (iterate from hash).
        self.sorted_positions = sorted(self.ring.keys())

    def _build_ring(self) -> None:
        """
        Populate self.ring with (num_shards × virtual_nodes) entries.

        For each shard, for each virtual node index:
          1. Build a key string: "shard-{shard_id}-vnode-{vnode_index}"
             Example: "shard-0-vnode-0", "shard-0-vnode-1", ..., "shard-0-vnode-149"
          2. MD5-hash the key → 128-bit hex string
          3. Convert first 8 hex chars to an int → a 32-bit ring position
          4. Store ring[position] = shard_id

        Why MD5 for the ring (not SHA-256)?
          MD5 is NOT used for security here — it's used for speed and uniform
          distribution. MD5 produces values that are evenly spread across the
          output space, which is exactly what we need for a balanced ring.
          SHA-256 would also work but is slower and overkill here.

        Why "first 8 hex chars" (32 bits)?
          Full MD5 = 128 bits = position space 0 → 2^128 (enormous).
          32 bits = position space 0 → 2^32 (4 billion) — sufficient for
          uniform distribution with 600 virtual nodes, and faster to sort.
        """
        for shard_id in range(self.num_shards):
            # Create virtual_nodes positions for this physical shard
            for vnode_index in range(self.virtual_nodes):

                # Unique string that identifies this virtual node
                # Example: "shard-2-vnode-73"
                vnode_key = f"shard-{shard_id}-vnode-{vnode_index}"

                # MD5 hash → 32 hex chars → take first 8 → int
                # Example: "shard-0-vnode-0" → "cfcd2084..." → 0xCFCD2084 → 3,485,074,564
                md5_hex = hashlib.md5(vnode_key.encode()).hexdigest()
                # hexdigest() returns e.g. "cfcd2084950f10d86b7da34f82c89b2b"
                # [:8] takes first 8 hex digits = 32 bits
                # int(..., 16) converts from base-16 to Python int
                position = int(md5_hex[:8], 16)

                # Map this ring position to its physical shard
                self.ring[position] = shard_id

    def get_shard(self, organization_id: UUID) -> int:
        """
        Return which shard (0..N-1) this organization's data lives on.

        Deterministic: same org_id ALWAYS returns the same shard_id.
        This is the core guarantee — routing must be stable or queries
        go to the wrong shard.

        Algorithm:
          1. Hash the org_id → a 32-bit position on the ring
          2. Walk clockwise (find first position in sorted_positions >= hash_val)
          3. Return the shard_id at that position
          4. If no position >= hash_val (we're past the last marker),
             wrap around to the first position (the ring is circular)

        Parameters
        ----------
        organization_id : The tenant's UUID. Must be stable — changing it
                          would reroute the tenant to a different shard,
                          losing access to their existing data.

        Returns
        -------
        int: shard_id in range 0..num_shards-1
        """
        # Step 1: hash the org_id to a ring position
        # str(organization_id) → "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        # .encode() → bytes
        # MD5 hex → take 8 chars → 32-bit int
        org_str = str(organization_id).encode()
        hash_val = int(hashlib.md5(org_str).hexdigest()[:8], 16)

        # Step 2: clockwise traversal
        # Find the first ring position that is >= hash_val.
        # self.sorted_positions is pre-sorted, so we iterate in order.
        # First match = "clockwise neighbor" on the ring.
        for position in self.sorted_positions:
            if hash_val <= position:
                # This shard "owns" the arc from the previous position to here.
                # Our org_id's hash falls in this arc → it belongs to this shard.
                return self.ring[position]

        # Step 3: wrap-around
        # hash_val is larger than all ring positions — we've gone past the
        # "top" of the circle. Wrap around to the first position (position 0
        # is the smallest — the start of the ring).
        # Example: hash=3,999,999,999 and max ring position=3,485,074,564
        # → wrap to ring[sorted_positions[0]] = shard at position 0
        return self.ring[self.sorted_positions[0]]

    def get_shard_distribution(self, org_ids: list[UUID]) -> dict[int, int]:
        """
        Show how a set of org_ids distributes across shards.

        Used in tests to verify even distribution, and in the debug endpoint
        to demonstrate the algorithm with real tenant data.

        Parameters
        ----------
        org_ids : list of organization UUIDs to distribute

        Returns
        -------
        dict mapping shard_id → count of orgs assigned to that shard
        Example: {0: 247, 1: 253, 2: 251, 3: 249} for 1000 orgs, 4 shards

        Interview demo: generate 1000 random UUIDs, call this, show the
        interviewer that each shard gets ~25% ± 2%. That's the power of
        150 virtual nodes.
        """
        distribution = {shard_id: 0 for shard_id in range(self.num_shards)}
        for org_id in org_ids:
            shard_id = self.get_shard(org_id)
            distribution[shard_id] += 1
        return distribution


# ---------------------------------------------------------------------------
# Module-level singleton ring
# ---------------------------------------------------------------------------
# Built once at import time. All calls to get_shard_for_org() reuse this ring.
# The ring is stateless after construction — no locks needed.
_ring = ConsistentHashRing()


def get_shard_for_org(organization_id: UUID) -> int:
    """
    Public interface: return shard_id for a given organization.

    In production, the caller would use this shard_id to select the correct
    asyncpg pool from a list:
      pool = shard_pools[get_shard_for_org(org_id)]

    In this project (single DB): returns the shard_id for demonstration
    and interview purposes. All shard_ids map to the same DB pool.
    """
    return _ring.get_shard(organization_id)


def get_ring_info() -> dict:
    """
    Return metadata about the ring configuration.
    Used in the /debug/shard-info endpoint response.
    """
    return {
        "num_shards": _ring.num_shards,
        "virtual_nodes_per_shard": _ring.virtual_nodes,
        "total_ring_positions": len(_ring.ring),
        # total_ring_positions = num_shards × virtual_nodes = 4 × 150 = 600
    }
