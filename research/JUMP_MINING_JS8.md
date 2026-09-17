# Jump research — Jane Street RTMDF 8th place (repo: 1821746019/Jane-Street-Real-Time-Market-Data-Forecasting-8th)
Mined 2026-09-17 as part of Diego's jump-research track. Source: solution.md (GitHub), since Kaggle discussion pages are not API-fetchable (403 with our token; web_fetch blocked).

## Their stack
- Time-series CV, val size = public test size; extra "gap" fold (200-day gap) to simulate private shift. Correlated well with LB.
- GRU over one-day sequences of (symbol-day) rows; 1-layer GRU + 2 linear beat 3-layer GRU. MLP/transformers/cross-symbol attention did NOT work.
- Aux targets: rolling averages of the main target at other horizons (constructed from other responders). +0.001-0.003.
- Market averages per date/time + rolling stats per symbol: +0.002.
- time_id as input feature (regime indicator).
- ONLINE LEARNING during inference: one SGD step per new day with real targets: +0.008 CV. Their biggest single gain.
- 6-model seed ensemble: 0.0105 -> 0.0112 LB.

## Applicability to MSCapital (CSV submission, no inference API)
- ONLINE LEARNING: NOT POSSIBLE. MSCapital takes a one-shot submission.csv; no targets are revealed during inference. Dead.
- Aux targets from future market bars: DEAD. Verified 2026-09-17: train/market.feather has no rows with seconds_before_predict < 0 (min = 0.0 over 30M rows). The label comes from data we never see; no multi-horizon aux targets can be constructed.
- Cross-sectional market averages (per absolute timestamp): DEAD. Samples have no shared absolute clock (only month + seconds_before_predict); cannot align across samples.
- time_id/regime input feature: RISKY. Adversarial AUC late-vs-test 0.657-0.695 is driven by exactly these regime features; month as input would be OOD on test months (unseen values -> blind extrapolation). Not tried; low priority.
- GRU architecture lesson (1-layer GRU + MLP head > deep GRU): APPLIED to seq3 design choice when writing the val kernel.
- Their "gap fold" validation: PARTIALLY APPLIED. Our shift sim (tr<=50 -> 51-60) is the same idea; it does not predict test transfer here (documented in LOG).

## Verdict
The big JS8 levers are unavailable in this competition's format. Confirms seq3 (order-flow channels) as the live candidate from this research line.
