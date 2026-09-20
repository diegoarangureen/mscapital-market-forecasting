# ZWQ1037 public toolkit mining (Sep 20, 2026) - CC BY 4.0

Source: public dataset zwq1037/mscapital-financial-market-forecasting-toolkit (imported
to reference/zwq1037-toolkit/ with attribution) + his public kernel
zwq1037/factorized-transformer-temporal3fold-lb0145 (LB 0.145 component).

## His final state (from PROJECT_CLOSEOUT.md, dated Sep 20)
- Best public LB: 0.152. Final submission = blend: public model blocks (bestwater GPU TabM
  21%, bestwater TPU TabM 5.7%, YangQ public preds 20.4%) 47.1% + own models 52.9%
  (Factorized Transformer raw 40% / market-conditioned event-residual Transformer 60%
  of the transformer slot), soft-gated on realized_volatility_60.
- COMPLIANCE NOTE: the 47% public-prediction blending is exactly what our rules forbid
  (yangq369 trap). We do NOT copy it. His own-models-only level is ~0.145 (kernel title).
- His own conclusion: bottleneck is new SINGLE-MODEL signal, not blending. Matches iamltpn (5th).

## Confirmed externally (independent of us)
1. XS per-month ranks: his XS40 (20 rank + 20 zscore cols) = NET-NEGATIVE on full 62-70
   (-0.00016), gains only when month 66 excluded. Month 66 is anomalous (cos 0.234 vs
   0.12-0.15 elsewhere). Same toxicity as our XS75. XS stays dead.
2. OFI is in his TOP-20 features (ofi_1_20_mean, ofi_1_last) - the X26 axis (building now)
   is the right one.
3. Aggressive-trade imbalance + depth-crossed pressure: his strongest feature family
   (EXP-TREE-020/023: +0.013/+0.015 over 132f baseline, 11/11 months). Top feature:
   trade_volume_imbalance_20 (last-20s aggressive buy/sell volume imbalance). Depth crosses
   (aggressive volume / visible L1 depth log1p, imbalance-vs-book-imbalance interactions)
   are NOT in our stack. -> X27 candidate.
4. Dead lines confirmed by his strict panel: Patch-Transformer-GRU, Axial-LOB, TSMixer, TCN,
   linear sequence, last-readout, Deep3 Transformer, Transformer EMA (0% blend weight),
   within-month rank targets, tree features past ~307, LightGBM tree-count tuning.
5. Evaluation protocol (best-in-class, adopt): overall + 62-70-no66 + 67-70 slices,
   per-month cosine, monthly std, worst month, q25, LOMO (leave-one-month-out) minimum.
   Promotion gate: must improve selection AND forward windows AND pass monthly stability.
   Our current val harness reports only 61-70 + 66-70 whole-vector - upgrade it.
6. His Factorized Transformer recipe (0.145 component): 3 stream grids (market 200x12,
   tx 60x7, order 60x10), per-stream projection + pos/table embeddings, 2 ConvBlocks,
   2-layer TransformerEncoder (d=96), attention pooling, 379 static features merged at head.
   3 temporal cutoffs (52/57/62), train-all-up-to-cutoff, fixed 4 epochs, no early stopping,
   0.35 SmoothL1 + 0.65 centered whole-batch cosine, target / train-std.
7. His robustness fix EXP053R: recompute m_rv denominator exactly from raw sequence -
   implementation-exactness of features mattered more than new features at his level.

## Updated experiment queue
1. X26 (OFI/VPIN) - building now, gate +0.004.
2. Val harness upgrade: multi-window + per-month + LOMO panel (CPU, no quota).
3. X27: aggressive-trade 20s/40s/60s imbalances + depth-crossed pressure + trade freshness
   (his proven family; we have partial coverage via tx_imb but NOT depth crosses).
4. If features keep flat: Factorized-Transformer-style multi-stream sequence model is the
   only architecture with public evidence of LB transfer in THIS comp (his 0.145 kernel);
   our GRU failure was a different (weaker) design. Reconsider sequence line with his recipe
   as the reference implementation (public, inspectable).
