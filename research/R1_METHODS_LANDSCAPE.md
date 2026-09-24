# R1 — Methods landscape (research phase, Sep 24). Sources: public competition discussions, read Sep 24 06:48-06:55.

## A. The plateau is method, NOT data
- Maaax (rank 1, 0.180), asked publicly whether >0.169 uses external data: "No, I haven't used any external data or externally pretrained models - only the competition data." (discussion/742707)
- ra1nac1d (12th) believes ceiling ~0.16; Maaax's 0.180 on competition data alone disproves a hard data ceiling.
- Host (Rib~): top-10 must submit AI-reproducible prompts/solutions for audit; non-reproducible -> next in line.
- ZWQ1037 toolkit docs and bestwater's posts remain CC BY 4.0 / public. bestwater's 462-feature dataset is PRIVATE (off-limits) - but his METHOD posts are public and mined here.

## B. Architecture evidence (public LB, single models)
| model | who | LB single | notes |
|---|---|---|---|
| GRU + window-stats fusion | Youler (7th) | 0.143 | two AIs built two feature sets; 10 seeds |
| GRU-repr + tree (2-stage) | Youler | 0.142 | freeze GRU repr, then tree on repr + window stats |
| TabM (cos, 3seed) | bestwater | 0.142 (GPU) / 0.142 (TPU) | TabM drift 10seed 0.141 |
| RealMLP / PTK | bestwater | 0.137 | ours v13 (RealMLP 450f): 0.139 |
| Transformer / FTT | bestwater | 0.120-0.129 | deep variants LOSE LB |
| LightGBM/CatBoost 90f | bestwater | 0.119-0.131 | 600+ features: CV 0.1409 but LB 0.116 (CV overfit) |
- Ensembles: bestwater production blend 0.148 (incl. 40% external-model blends); Youler grid blend 0.147. Ensemble gain ~+0.005-0.01 over best single.
- Youler: full-data training gives no gain (matches bestwater's refit-on-full finding and our protocol: holdout + early stop wins; refit-on-full 0.142 -> 0.137).

## C. Protocol confirmations (independent, bestwater + Youler)
- XS/cross-sectional features: TabM xs689 CV 0.1510 (his max) -> LB 0.136. THIRD independent confirmation of our v14 finding (XS gains CV, loses LB). Absolute per-sample features only for submissions.
- Month 66 is anomalous (too easy, inflates CV) - bestwater excludes it too; matches our ZWQ-adopted protocol.
- bestwater's stricter CV: rank-normalize target within month to [-1,1], purged folds, validate on concatenated 67-70 (CV(rn67-70)). Under raw-target 40-70 his RealMLP shows 0.162; under rn67-70 0.1314. LB/CV ratio > 1 for cos-trained models (LB 0.142 vs CV 0.1226-0.131) - rank-norm CV UNDERESTIMATES LB but tracks it better.
- bestwater's "distribution alignment experiment": CV 0.172 -> LB 0.120. Same local-gain-LB-loss trap as our X-series and ZWQ EXP026.

## D. Label-structure hypothesis (Xu Dian, discussion/742719, UNVALIDATED)
- Hypothesis: the label includes a term computed from features ~90-120 s into the future. Aux tasks on -300..0s market data peaked at a particular window. If true: features should target the 90-120s-ahead horizon (short-horizon order-flow persistence, microprice drift), and aux-task design can probe it cheaply.

## E. Online/self-training
- Timmy Juicehouse: open-source LightGBM offline 0.121 -> "online learning" 0.126 (+0.005 LB). Host: true online learning only works if the API serves labels; this comp is offline CSV -> the applicable version is pseudo-labeling / self-training on test features. Secondary direction.

## F. Host-recommended past comps (discussion/723768)
- Feature engineering: Optiver Realized Volatility Prediction, Optiver Trading at the Close (order-book feature canon).
- Models: CMI sensor data, NFL Big Data Bowl 2026, Child Mind sleep, LEAP ClimSim, geophysical waveform.

## G. Meta
- Top competitors run LLM-agent pipelines 24/7 with human direction (same shape as us). bestwater: hard gates must be scripted, not documented (matches our parent-authorization submission gate).
- Discussion list exhausted (single page): no public solution posts beyond these.

## Next-screen proposal (X30) - evidence-ranked
1. **Optiver-style deep book/trade feature pack v2** (direction 1, biggest gap): multi-level OFI (L1-L5), trade-sign autocorrelation, realized-vol estimators (Parkinson/Garman-Klass on bars), order arrival intensity, book slope/curvature beyond L2, queue imbalance dynamics. Build kernel-side from raw streams (X29 pattern), screen on champion RealMLP 5-fold panel. Rationale: Maaax proves headroom exists on competition data; bestwater's private 462 features are his edge; Optiver comps are the host-pointed canon.
2. **TabM port** (architecture, bestwater's best single 0.142 vs our 0.139): moderate build cost on TPU; screen vs champion panel.
3. **GRU two-stage** (Youler's 0.143 single / 0.147 blend): sequence side on raw streams; bigger build, keep as follow-up.
4. Aux-task probe of the 90-120s label hypothesis (cheap, informs 1).
