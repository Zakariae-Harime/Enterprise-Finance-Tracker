"""
Debug Routes — Development & Interview Demo Endpoints

These routes are for demonstrating system design concepts live.
In production: mount only if DEBUG=true env var is set (see main.py).

Endpoints:
  GET /debug/shard-info   → shows which shard the current user's org lives on
  GET /debug/ring-stats   → shows the full ring configuration + distribution demo
"""
from fastapi import APIRouter, Depends
from uuid import uuid4, UUID
from src.auth.dependencies import get_current_user, UserContext
from src.infrastructure.sharding import get_shard_for_org, get_ring_info, ConsistentHashRing

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
)


@router.get("/shard-info")
async def shard_info(
    current_user: UserContext = Depends(get_current_user),
):
    """
    Show which database shard this user's organization would live on.

    INTERVIEW DEMO SCRIPT
    ----------------------
    1. Start the API and authenticate (get a JWT)
    2. Hit this endpoint: GET /api/v1/debug/shard-info
    3. Show the interviewer:
         {
           "organization_id": "00000000-0000-0000-0000-000000000001",
           "shard_id": 2,
           "total_shards": 4,
           "virtual_nodes_per_shard": 150,
           "explanation": "All data for org 00000000... lives on shard 2.
                           In production, shard 2 = a dedicated PostgreSQL instance."
         }
    4. Explain: "The hash ring maps this UUID to position X on the ring.
       Walking clockwise, the first virtual node we hit belongs to shard 2.
       This mapping is deterministic — the same org_id always gets shard 2.
       If we add a 5th shard, only ~25% of orgs move. The rest stay on their
       current shard with no migration needed."

    Why this endpoint exists:
      - Makes the abstract algorithm tangible (real UUID, real shard number)
      - Shows the interviewer you can explain CS theory with working code
      - Proves the system is actually wired, not just described
    """
    org_id = current_user.organization_id

    # get_shard_for_org() runs the MD5 hash → clockwise ring walk algorithm.
    # Always deterministic: same org_id → same shard_id, every time.
    shard_id = get_shard_for_org(org_id)

    ring = get_ring_info()

    return {
        "organization_id": str(org_id),

        # The shard this org's data lives on (0-indexed).
        # In production: maps to a specific DATABASE_URL.
        "shard_id": shard_id,

        # Total physical shards in the ring.
        "total_shards": ring["num_shards"],

        # Each physical shard has this many positions on the ring.
        # More virtual nodes = more even distribution.
        "virtual_nodes_per_shard": ring["virtual_nodes_per_shard"],

        # Total positions on the ring = num_shards × virtual_nodes.
        # Example: 4 × 150 = 600 positions
        "total_ring_positions": ring["total_ring_positions"],

        "explanation": (
            f"Organization {org_id} is deterministically mapped to shard {shard_id}. "
            f"All its events, expenses, accounts, and budgets would live on "
            f"the shard {shard_id} database in production. "
            f"Adding a 5th shard would only move ~{100 // ring['num_shards']}% "
            f"of tenants — not all of them."
        ),
    }


@router.get("/ring-stats")
async def ring_stats(
    current_user: UserContext = Depends(get_current_user),
):
    """
    Show ring configuration and demonstrate distribution with 1000 random orgs.

    INTERVIEW DEMO SCRIPT
    ----------------------
    Hit this endpoint to show:
      1. Ring has 600 total positions (4 shards × 150 virtual nodes)
      2. 1000 random org_ids distribute ~evenly: each shard gets ~250 ± 10
      3. Standard deviation across shards is typically < 2%

    This proves the claim "virtual nodes give near-perfect load balance"
    with real numbers, not just theory.

    WHY 1000 RANDOM ORGS?
    ----------------------
    The distribution is a statistical property — you need enough samples
    to see it. 1000 is a good demo size:
      - Large enough to show near-even distribution
      - Fast enough to compute in <10ms
      - Realistic scale for a growing SaaS (1000 enterprise tenants)
    """
    ring_info = get_ring_info()

    # Generate 1000 random UUIDs to simulate 1000 hypothetical tenants.
    # uuid4() generates a random UUID each time — no pattern, just randomness.
    # This simulates "if we had 1000 real enterprise tenants, how would
    # they distribute across our shards?"
    sample_orgs = [uuid4() for _ in range(1000)]

    # Build a ring instance and calculate distribution.
    # We use the module-level ring config (4 shards, 150 virtual nodes).
    ring = ConsistentHashRing(
        num_shards=ring_info["num_shards"],
        virtual_nodes=ring_info["virtual_nodes_per_shard"],
    )
    distribution = ring.get_shard_distribution(sample_orgs)

    # Calculate what percentage of orgs each shard holds.
    # Perfect balance = 25.0% each for 4 shards.
    percentages = {
        f"shard_{shard_id}": {
            "count": count,
            "percentage": round(count / len(sample_orgs) * 100, 1),
        }
        for shard_id, count in distribution.items()
    }

    # Standard deviation measures how far each shard deviates from ideal.
    # Lower is better. With 150 virtual nodes, expect < 2%.
    counts = list(distribution.values())
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    std_dev_pct = round((variance ** 0.5) / len(sample_orgs) * 100, 2)

    return {
        "ring_configuration": ring_info,
        "demo_sample_size": len(sample_orgs),
        "distribution": percentages,
        "std_deviation_pct": std_dev_pct,
        "balance_assessment": (
            "excellent (< 2%)" if std_dev_pct < 2
            else "good (2–5%)" if std_dev_pct < 5
            else "poor (> 5% — increase virtual nodes)"
        ),
        "resharding_impact": {
            "current_shards": ring_info["num_shards"],
            "if_shard_added": ring_info["num_shards"] + 1,
            "pct_orgs_that_must_migrate": round(100 / (ring_info["num_shards"] + 1), 1),
            "pct_orgs_unaffected": round(100 - 100 / (ring_info["num_shards"] + 1), 1),
            "vs_modulo_sharding": "100% must migrate with modulo (hash % N)",
        },
    }
