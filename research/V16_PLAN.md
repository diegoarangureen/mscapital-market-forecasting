# v16 candidate plan (draft, Sep 21 2026) — NO submission without explicit owner authorization

Champion: v13 = LB 0.139 (rank ~125/225). v15 (25-model) = 0.138 flat.
OOF reference (kfold450 5-seed v2): 61-70 = 0.16827, 66-70 = 0.17522. Fold5 seed2026 = 0.17009.

## Hard gates (parent rules)
- val >= 0.130 clear AND robustness verified.
- No submissions on changes below LB resolution (~0.002) unless strong forward-sim evidence.
- Quota: 5 submissions/day, resets ~02:00 CEST.
- Every submission requires explicit owner/parent authorization. Never self-authorize.

## Validation protocol for any v16 candidate (adopted from ZWQ1037, stricter than before)
- Selection window and forward-confirmation window must be DISJOINT:
  select on 62-65, confirm on 67-70, exclude month 66 (anomalous) from scoring.
- Gate: +0.004 vs champion same-window OOF (val noise floor), plus panel:
  all primary deltas >= 0, LOMO min >= -0.001 (src/kaggle/panel.py).

## Candidate paths, ranked (re-ranked Sep 24 after X28/X29 closures)
1. **X30: Optiver-style proprietary feature pack v2** (ACTIVE, parent green light Sep 24 06:51).
   Multi-level OFI (Cont-Kukanov-Stoikov), trade-sign autocorrelation, realized-vol
   estimators, order arrival/cancel intensity, queue-imbalance dynamics, microprice
   drift. Kernel-side build from raw streams (X29 pattern), screen on champion RealMLP
   5-fold panel. Evidence: R1_METHODS_LANDSCAPE.md - Maaax (1st, 0.180) uses competition
   data only, so headroom exists; host-pointed canon is Optiver; bestwater's private
   462 features are his edge. Encoding facts: order_action 0=NEW 1=CANCEL; side from
   price vs mid; market spans ~600s, order/transaction 60s.
2. **TabM port** (bestwater's best public single 0.142 vs our 0.139): architecture
   screen vs champion panel. Moderate TPU build.
3. **GRU two-stage** (Youler 0.143 single / 0.147 blend): sequence repr + tree.
   Bigger build; only if 1-2 stall.
4. **Aux-task probe of the 90-120s label-horizon hypothesis** (Xu Dian, unvalidated):
   cheap, informs feature-horizon design for #1.
CLOSED this cycle: TPU replication (done, port validated), X28a/b/c (all flat),
X29 price=0 masking (flat, 4/5 folds within +-0.0002 of baseline).

## Non-candidates (closed, evidence in LOG/EXPERIMENTS)
- More equal-weight aggregation (v15 ceiling), loss balance (X28a), TabM arch,
  XS/transductive features (v14 + 2 independent confirmations), new public-data
  feature families (X23-X26, RS1/RS2, EXP067/068).

## Update Sep 21 19:45 — consolidated evidence from public-source mining

**TPU status**: v5e-8 CONFIRMED 18:43 (Persona verification works; probe queued
~7h for capacity — expect multi-hour TPU queue times for all sessions).
Replication (tpu455repl, seed2026 fold5 vs 0.170090) running; full kfold TPU
launches automatically on fidelity confirm.

**X28c grounding strengthened**: ZWQ EXP-TREE-017-019 tested within-month
percentile-rank features (+0.0048 formal cosine) but rejected them as
transductive and non-replicating. His own suggested causal fix (train-fitted
empirical CDF mapping per column, fixed for future months) IS our X28c
(EXP-TABM-011 evidence: +0.001 both windows, worst-month +0.01). X28c stays
#2 in the screen queue with independent motivation.

**New open item — empty book levels**: order-book `price=0` marks an EMPTY
level, not a price (senanuretin, verified by book balance). Our X21/X22
builders do not mask zero prices; microprice/spread features inherit sign
artifacts on empty levels. Action when a feature screen opens with budget:
(1) quantify price==0 frequency in market/order streams, (2) A/B masked
variants of affected X21/X22 features. Encoding facts for any future
order-flow family: order_action 0=NEW, 1=CANCEL; side recoverable from
price vs mid; market spans ~600s, order/transaction 60s.

**Sequence line (only if single-model line exhausts)**: ZWQ's best owned
component was the market-conditioned event-residual transformer (60% of his
transformer slot; raw40/event60 gave +0.0022 pooled / +0.0025 forward, 4/4
months positive — but only matched 0.152 on LB as a blend). His factorized
transformer source (public kernel, LB 0.145 standalone) fetched for
reference. His closeout conclusion matches ours: the bottleneck is absence
of new single-model signal on public data.

**Validation-period difficulty** (senanuretin): monthly target vol swings
2.69x; a frozen model scores 0.117-0.148 depending on the block. Any v16
candidate comparison must use identical windows vs champion, and the 62-65
select / 67-70 confirm split already guards this. Her calibrated adversarial
check (block-to-block, not pooled) found test period NO more drifted than
ordinary train-block distance — mild counter-evidence to pure-drift
explanations of the local/LB gap.
