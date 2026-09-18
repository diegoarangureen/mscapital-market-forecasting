# Why XS75 gained val but lost LB (Sept 18, post-v14)

## Facts
- x530 val (tr<=60, va 61-70): 0.173447 vs x455 0.167133 (+0.0063). Late window also up.
- kfold530 OOF: 0.170137/0.178068 vs kfold455 0.167809/0.174533 (+0.0023).
- LB: v13 (455f) 0.139, v14 (530f) 0.135. XS75 reversed sign on LB.

## What XS75 is
Within-month percentile ranks (top30 LGBM-gain features), within-month z-scores (top15),
cross-feature ranks (top30). Train: computed within each train month. Test: GLOBAL over the
whole test set (no month labels exist for test rows).

## Hypotheses for the transfer failure
1. **Definition mismatch train vs test.** Train XS is within-month (each month ranked against
   itself); test XS is global across all test months. If the public test spans multiple months
   with drift, a feature that means "rank inside my month" in train means "rank inside the whole
   test blob" at inference - systematically different object. This alone can flip sign.
2. **Regime sensitivity.** Cross-sectional ranks are relative measures; under the diffuse
   distribution shift we measured (adversarial AUC ~0.78 after pruning), relative position
   features shift differently than absolute ones, and RealMLP's PBLD embeddings may overfit
   the train-specific rank distribution.
3. **Interaction with advdrop.** Our base 298 is adversarially pruned for stability; adding 75
   unpruned cross-sectional features reintroduces shift through the back door. The adv AUC
   of the 530f set was never re-measured - worth checking when GPU returns.

## Testable follow-ups (queued for GPU refresh)
- x603-val: 455f + X23 (order-flow stream aggregates, ABSOLUTE features, no cross-sectional
  component) - isolates whether the problem is cross-sectionality or any new feature layer.
- Adversarial AUC of 455f vs 530f to confirm hypothesis 3.
- If XS ever retried: compute test XS within GUESSED test months (e.g. via seconds_before_predict
  clustering) instead of global, or drop the layer entirely.

## Standing rule update
Features with cross-sectional or transductive computation are LB-risk; val/OOF evidence is not
sufficient for them. Absolute, per-sample features only for submissions unless LB-tested.
