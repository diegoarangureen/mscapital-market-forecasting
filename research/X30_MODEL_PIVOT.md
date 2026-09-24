# Model-class pivot plan (prepared 2026-09-24 20:40, for pairrep verdict)

## Evidence: model class is not the bottleneck
- GRU on streams: DEAD, three independent builds (Sep 16-17): seq1 raw 0.0162, seq2 120x8 market-only 0.0197, seq3 120x12 +order-flow 0.0364. Discard bar was 0.10; tabular RealMLP val 0.149. Order-flow channels nearly doubled seq signal (0.0197->0.0364) but ceiling ~25% of tabular.
- TabM@455f: FLAT, closed Sep 20 (fold4 0.14442, fold5 0.15720, OOF 0.15206 vs bar 0.165). bestwater's TabM edge (pool 0.1638 @689f per his own kernel docstrings) comes from his 462 private features, NOT the architecture. Blend with RealMLP: r=0.83, forward-sim +0.0010 max.
- RealMLP 450f champion: validated local optimum (every config neighbor flat or worse).

## The one model lever still open
The public best-single GRU 0.143 (discussion/733271). Our dead GRUs were RAW-STREAM GRUs; the public 0.143 may use a materially different input design (sequence construction, target, ensembling). Before any port: re-read discussion/733271 and extract the exact recipe. Port only if its input design differs from our three dead builds.

## If pairrep says X30 FLAT — pivot order
1. Re-read 733271 (15 min, no compute). If the public GRU is raw-stream-like -> model class fully closed, no TPU spend; pivot fully to feature/signal research (X31: 600s market ticker aggregates, cross-sample structure without XS normalization).
2. If its design is genuinely different -> port as screen on the same 5-fold panel protocol. Cost estimate: our seq GRU vals ran ~30 min on GPU (hid 96, BS 1024, 16 epochs, 1.26M x 120 x 12); a faithful port screen is ~1-2h GPU (quota ~29 min/day left, so realistically next-day GPU or a TPU port at ~1.5-2.5h; TPU quota after pairrep ~2.2h of 20h weekly).
3. Submissions unchanged: only with explicit parent authorization, val >= 0.130 clear + robustness verified.

## If pairrep says X30 SIGNAL
Ablate X30 by family (per X30_CONTINGENCY.md), extend the carrying family, then forward-sim gate before any submission ask.
