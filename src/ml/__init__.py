# src/ml/ — Machine Learning pipeline for transaction categorization
#
# Three-layer architecture (fastest to slowest, cheapest to most powerful):
#   Layer 1 — Rules:   O(1) dictionary lookup. 0ms. REMA 1000 → groceries.
#   Layer 2 — TF-IDF:  Statistical ML. 1ms. No GPU needed.
#   Layer 3 — NB-BERT: Norwegian BERT deep learning. 8ms. Understands context.
#
# Each layer only activates when the previous layer is not confident enough.
# 80% of transactions are caught by Layer 1 (known Norwegian merchants).
# 15% by Layer 2 (new merchants with recognizable keywords).
# 5% by Layer 3 (ambiguous, context-dependent).
