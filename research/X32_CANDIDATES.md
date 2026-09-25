# X32 candidate inventory (FLOW signal mining, written 2026-09-25)
Composition rule: X32 = (X31 features that screen positive) + (new candidates
below, prioritized). Do NOT build before the X31 3-arm verdict - flat X31
features get dropped, and the screen tells us which DIRECTIONS pay.
Hard rules: absolute per-sample only (XS trap 3x); nonlinear/distributional
variants preferred (linear combos of existing features are free for RealMLP);
every candidate must be computable with bincount/np.add.at accumulators in a
single ziter pass; synthetic smoke test incl. degenerate sample before push.

## A. Untapped in the 60s flow window (highest prior - FLOW is the confirmed carrier)
1. Trade aggressor run-length stats: max/mean consecutive same-sign trades per
   sample. Momentum bursts; zero overlap with X30/X31. Cheap: sorted tx stream.
2. Volume-weighted trade-sign autocorr (lag1): X30 tsign_ac1 is count-based;
   weight by volume - big aggressive sequences matter more.
3. Order-size distribution per side: mean/max NEW order volume bid/ask +
   max/mean ratio (whale detection). Complements X31 ord_vnet (sums).
4. Arrival intensity shift: ord_add_rate in [0,10) vs [10,60) - the order-flow
   analog of X30's flow_shift (which was trades). X31 has gaps, not this.
5. Cancel waves: fraction of cancels arriving within 1s of a previous cancel
   on the same side. Quote-fading cascades; distributional.
6. QI lead-lag response: per-sample corr(qi1_t, next mid return) over the 60s
   window. A direct "does the book predict the next tick" measure; needs one
   sorted pass, accumulate sum(x*y), sum(x), sum(y), sum(x2), sum(y2).

## B. 600s context flow (market stream spans 600s; X30/X31 only used <60s for flow)
7. Context OFI: OFI L1/L2 summed over [60,600) - the pre-window flow state the
   60s flow sits on top of. Same accumulator pattern, different time mask.
8. Context QI: mean qi1 over [60,600) and its drift into the last 60s
   (qi1_mean_[60,600) vs qi1_mean_[0,60)) - regime shift measure.
9. Context depth: mean L1/L2 depth over [60,600) as normalization-free levels
   (log) - liquidity regime, interacts with flow meaning.

## C. Cross-feature nonlinear ratios (cheap, only if X31 shows ratios pay)
10. OFI per unit spread (X31 note had it, dropped for size): ofi1_all /
    spread_rms - flow intensity adjusted for book tightness.
11. Kyle-lambda asymmetry: separate lambda for buy-initiated vs sell-initiated
    trades (X31 kyle is signed-pooled).

## D. Explicitly rejected
- Cross-sectional anything (3x LB trap).
- Deeper model classes (closed: GRU x3, TabM flat, all public archs < 0.139).
- More RV estimators (VOL ablation flat - that family is exhausted at 60s).
- Linear recombinations of existing per-bucket sums (free for the NN).

## Selection rule after X31 verdict
- If X31 delta > +0.002 consistent: keep its winners, add A1-A3 (cheapest,
  most orthogonal) -> X32 ~12-18f, same 3-arm screen pattern.
- If X31 flat: stop extending FLOW at this horizon; pivot to B (600s context)
  with a 2-arm screen; FLOW stays as X30-only contribution.
