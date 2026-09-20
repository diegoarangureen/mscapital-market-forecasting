# Open-source sweep (Sep 20, 2026) - requested by Diego

## Sources mined
- bestwater's experiment-summary post (6d ago): https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/discussion/733271
- Comments on it from iamltpn (5th), MAIDANG (6th), sichao shen (9th), yunsuxiaozi (50th)
- GitHub open solutions: Jane Street 2024 (8th place repo + 4 others), stefan-jansen ML4T microstructure lib, orderflow-metrics, VPIN_HFT, lob-regime-scanner
- bestwater_tabm references (already in repo)

## What the public record says (validated against our findings)
1. Sequence/Transformer models: bestwater's Transformer CV 0.1549 vs LGB 0.1409, but LB 0.120 vs 0.119 - CV gains do NOT transfer. External confirmation that our sequence line is dead for everyone, not just us.
2. More features past a point = CV overfitting (his 600+ features: CV up, LB down). Matches our 8/8 flat stacking vals.
3. Post-processing (standardize, center+tanh): CV +0.002, LB flat. Matches our calibration-ceiling finding exactly.
4. His distribution-alignment experiment: CV 0.172, LB 0.120. Same trap as our XS75 (val up, LB down). Alignment/transduction is LB-toxic across the board.
5. Sanity checks from his post: all-ones pred = -0.007, all-minus-ones = +0.007 - target vector has a slightly negative mean; metric asymmetry confirmed public.

## Top-10 player advice (public, on the record)
- iamltpn (5th): CV and LB are consistent below LB ~0.150; above that they diverge. Priorities: (1) reliable offline validation, (2) fast iteration volume, (3) improve the SINGLE model, not fusion.
- MAIDANG (6th): single metric -> CV overfitting; use multiple generalization metrics offline for robust experiment selection; fusion helps but more models is not better.
- Both endorse what we already do (kfold-OOF protocol, equal-mean moderate ensemble). The path upward they point to is single-model strength + trustworthy validation.

## Coverage check: our 246 base features vs classic open-source microstructure formulas
- HAVE: order imbalance (19 cols), spreads (12), multi-horizon returns/momentum (12), volatility family (15), trade imbalance (tx_imb*), order new/cancel imbalance, micro-price approx (X7:b*_micro), Kyle lambda + price impact (X25, tested FLAT as x472).
- MISSING (structurally different axis = event/volume-synchronized, our whole stack is time-synchronized):
  * OFI (Cont, Kukanov, Stoikov order flow imbalance) on L1/L2 1s snapshots - event-driven bid/ask price+volume change accumulation
  * OFI acceleration (second derivative - bestwater lists it explicitly)
  * VPIN (Easley/Lopez de Prado volume-synchronized PIN) - volume-clock bucketing of signed flow
  * Roll implied spread, Amihud illiquidity (low prior: likely correlated with existing vol/spread cols)

## Decision: next experiment = X26
Volume/event-synchronized microstructure: OFI + OFI-acceleration + VPIN at multiple volume-clock scales (+ Roll/Amihud as cheap fillers). Build as CPU kernel (same pattern as X25), val as 455f+X26 controlled A/B vs champion 0.1671. Discards if delta < +0.004 (noise floor). This is the last untried feature AXIS; if flat, feature engineering is exhausted and the remaining lever is single-model architecture/training.
