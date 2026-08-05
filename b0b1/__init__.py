"""B0/B1 flow-matching generative policies for RoboCasa.

From-scratch DiT-based action/intent generation on frozen VL features.
Supports two arms:
  B0: p(A | o, l) — action-only generative BC
  B1: p(I | o, l) · p(A | o, l, I) — intent-conditioned generative BC with T-mask
"""
