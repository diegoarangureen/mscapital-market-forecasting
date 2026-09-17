# MSCapital discussion + public kernel mining (2026-09-17, via cloud browser)
Competition: 22 days to go (ends ~Oct 9). Rules reaffirmed by host: organizer data free to use/process; external data/models strictly prohibited.

## Leaderboard context (from posters' ranks)
- Top-10 final line estimated 0.16-0.17 (sichao shen 6th: 0.16; Yone: ~0.17). Current gate10 0.160, top1 0.172.
- Zigui Wang (20th): feels public-LB ceiling for local-val approaches is ~0.16.

## bestwater (25th) — full experiment dump (topic 733271)
Single models (his CV = purged 5-fold, per-month rank-normalized targets, eval on months 67-70):
- TabM multi-head (689f: 462 own + 152 yunsuxiaozi-0726 + 75 XS): LB 0.142 (GPU and TPU), drift-10seed 0.141
- ResMLP 0.139; RealMLP/PTK 689f: LB 0.137; CNN-Transformer 0.136 (low test-corr 0.861 -> diversity)
- LightGBM xs689 0.131; CatBoost 0.130
- Transformer: CV 0.1549 -> LB 0.120 (CV-LB inversion, same as ours)
- Distribution alignment: CV 0.172 -> LB 0.120 (FAILED - matches our adv_iter finding that shift is diffuse)
- Production blend incl. EXTERNAL public-kernel preds (YQ=yangq369 etc.): 0.147-0.148 -> we don't do this (yangq369 trap)
Training paradigm (his table 2): holdout+ES TabM 0.142 vs refit-full 0.137; ResMLP holdout 0.129 vs refit 0.100. Cutting early months hurts (0.137->0.124).
NOTE: our v12 holdout+ES RealMLP was flat (0.124) - but his TabM test pred = 5 purged fold-models x 3 seeds averaged, on 689 features. Different feature set + multi-model averaging; our single-model test doesn't close the question.

### bestwater TabM kernel (Apache 2.0, public score 0.142, source saved research/reference/bestwater_tabm_cos689.py)
- Arch: shared MLP (3x Linear512-GELU-BN-DO0.15 + 256 layer) with 64 averaged linear heads (NOT true TabM BatchEnsemble)
- Cosine loss on RAW target (+0.020 vs RN+MSE in his tests); standardize by train<=62 stats
- 5 purged folds (val 40-44/50-54/55-59/60-64/65-70, train <=37/47/52/57/62), ES patience 15, 60 epochs, bs 2048, 3 seeds
- Test pred = mean over 15 fold-seed models; no clipping/zeroing
- XS 75 features: within-month percentile-rank of LGB-top30, within-month z of top15, cross-feature rank of top30. TEST versions use test-global rank/z (transductive - and it HELPS him; nuance vs UnseenAnchor's "transductive norm" negative)

## Youler (10th): GRU + window stats, two-stage (freeze GRU repr -> tree on repr+stats): GRU1 10seed CV 0.1766 -> PB 0.143; GRU+Tree 0.142; grid fusion 0.147. So sequence models CAN reach 0.14+ on LB - but via window-stat fusion and multi-seed averaging, not bare GRU (our seq1-3 bare-GRU builds: 0.016-0.036 val).
## iamltpn (11th): CV-LB consistent below 0.150 for him; focus on reliable offline val + single-model strength over fusion.
## yunsuxiaozi (76th, public kernel author): RealMLP valid 0.141 -> LB 0.131 (gap 0.010 on row-split val). Ensemble of 3 feature/model variants 0.135->0.141.
## heng (35th): tree, val last-20-months, CV 0.141-0.147 -> LB 0.123-0.130.
## Online learning thread (724670): no test labels in this comp; Timmy's "online learning" (+0.005: 0.121->0.126) was transductive test-feature usage, not supervised updates. Confirms JS8 online-learning lever is dead here.

## Actionables (priority order)
1. 0726 features (152, public kernel output yunsuxiaozi/rfmf-0726data): RUNNING - x455 val kernel (champion+152f), then mhmlp (bestwater-arch) val queued.
2. XS 75 within-month/transductive features: included in mhmlp val.
3. Multi-seed + fold-average inference for test (if val confirms): the 0.142 recipe is 15-model averaged holdout models.
4. yunsuxiaozi's other public kernels (rfmf-lightgbm etc.) for additional feature families.
