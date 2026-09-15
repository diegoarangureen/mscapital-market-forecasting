# Mining: UnseenAnchor/MSCapital-Kaggle (public competitor repo, found 2026-09-15)

Source: https://github.com/UnseenAnchor/MSCapital-Kaggle (public, pushed to 2026-08-27, ~19 Kaggle submissions documented).
A genuine competitor's full strategy + iteration log (Chinese). GPU rig (RTX 4070). Public peak 0.146 (Event256 Stack, ref 55601441). Plateaued at 0.145-0.146; they froze and stopped.

## Verified LB intel (their leaderboard snapshot 2026-08-18)
- Top1 public: 0.162; Top10 gate: ~0.153. Our 0.116 -> gap to top10 ~0.037.
- Their self-built hidden strength ~0.133-0.135; OOF 0.157; transfer gap OOF->public 0.012-0.022 (matches ours: val 0.137-0.141 -> LB 0.116).
- Single LGB caps ~0.116 public. All their gains came from unit-normalized diverse-architecture ensembles (LGB + GRU + CNN-Transformer grids + RealMLP). Sequence models need GPU -> not our path.

## CPU-feasible POSITIVE finding
- feat_microstructure_v3 (342 new feats -> 434-col LGB): gains on ALL folds: early 0.12747->0.13091, middle 0.13097->0.13701, late 0.13565->0.14638. Blend old25/new75 even better (0.13274/0.13815/0.14706).
- New families: L1/L2 mid, spread, depth imbalance, microprice, book slope; 10/30/60s signed trade volume AND signed amount; correctly signed new/cancel order pressure + side-specific cancellation imbalance; short/long-window deltas; cross-stream flow/book interactions.
- Their src/feat_microstructure_v3.py is public study material (no license -> reimplement ideas, don't copy verbatim).

## NEGATIVE results (do NOT burn CPU on these)
- Transductive input norm (test-stats normalization): FAILED late gate.
- Month-balanced sampling: FAILED (-0.008 to -0.02).
- Recency-weighted sampling: FAILED (Proxy -0.020).
- Domain-stable feature selection (target_gain/sqrt(domain_gain)): FAILED.
- LOO ridge meta-stack over fixed ensemble: <0.001, below gate; middle fold negative.
- target round-4: FAILED. RQ-KMeans target auxiliary: FAILED.
- Event-sequence family OOF ceiling ~0.12; CORAL/latent alignment failed; SSL domain adaptation failed: "shift is not fixable as simple marginal drift".
- Weight/meta re-search over existing predictions: exhausted. Only NEW independent information helps.

## Confirms our current direction
- Direct cosine optimization + de-mean/unit-norm at inference = their point 5. Our in-flight GBM cos-ES + MLPv2 cos-loss+EMA align.
- Multi-window mid price change rates (10/30/60/120/300s) = their #1 feature finding (val 0.1214->0.1360). Check our X-coverage.

## Trap to avoid (compliance)
- Their 0.144 came from blending 60% of a PUBLICLY POSTED LB-0.142 prediction file (yangq369/kaggle-lb0142-upload). Leaning on a public-LB artifact = external model + private-shakeup death. NOT our path; host prohibits external models.
