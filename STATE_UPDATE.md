# 2026-09-12 18:30 CEST - wipe #3 recovery (new agent)
- Sandbox wiped again at ~18:25; /tmp lost incl. running robustness check (eval_shift GBM part).
- Recovered ALL scripts from transcripts (this dir), repo recloned, data re-downloading.
- Candidate awaiting robustness+submission: 50/50 unit-blend of GBM v4 + MLP(121 feats: X2 + fp log-diff/log1p + X6).
  Val (61-70): blend 0.13126 centered (GBM 0.126436 recon / MLP 0.125204). Weight plateau smooth (0.4: 0.13115, 0.5: 0.13126, 0.6: 0.13090). corr(preds)=0.84.
- Plan: rebuild X2/X6/fp -> eval_x2base (expect 0.1258/0.1264) -> mlp.py (expect ~0.1313 blend) -> eval_shift.py + mlp_shift.py (robustness train0-50/val51-60) -> if holds, refit 0-70 both, blend, submit "GBM v4 + MLP blend".
- Submission quota: ~5/day, reset ~02:00 CEST.

## 2026-09-13 20:27 CEST - rebuilt + reproduced + robustness passed
- Rebuild: X2/X6/fp all rebuilt; baseline 3-seed 0.12469/0.12633 centered (matches predecessor 0.12644).
- Blend reproduced: MLP seedavg 0.12506 centered, corr 0.8373, blend w=0.5 val 0.131141 centered (smooth plateau).
- ROBUSTNESS (train 0-50, val 51-60): GBM 0.122894 centered, MLP 0.116482, blend 50/50 = 0.124462. Blend beats GBM on BOTH splits (+0.0048 main, +0.0016 shift). GO for submission.
- Running: mlp_epochs.py -> /tmp/work/mlp_best_epochs.json (log /tmp/mlp_epochs.log, marker EPOCHS_DONE).
- NEXT: run final_blend.py (GBM refit 0-70 @1.1x iters [110,116,77] + MLP refit @1.1x best epochs + 50/50 unit blend + nodata zero + clip + submission_blend.csv; marker FINAL_BLEND_DONE, log /tmp/final.log). Then submit via submit.py pattern (fix path to submission_blend.csv, description "GBM v4 + MLP blend"). Check quota first via list_submissions. Then leaderboard screenshot via cloud browser and report LB+rank to parent.

## 2026-09-13 21:07 CEST - SUBMITTED blend v1
- ref 56213569, public LB 0.113 (previous best 0.110, GBM v4). Description "GBM v4 + MLP blend".
- Refit preds saved: refit_gbm_test.npy, refit_mlp_test.npy (in src/recovered/). mlp_best_epochs {0:8,1:9,2:15}, x2 iters [110,116,77].
- Val->LB mapping: val 0.1274 -> LB 0.110; val 0.1311 -> LB 0.113. Gap remains large; next levers must close val-LB transfer.

## 2026-09-13 21:45 - structural finding: test regime = late months
- Per-month val cosine flat 0.10-0.17 except month 70 (hardest, 0.097-0.107). LB 0.113 ~= month-70 level: test (months 71+) follows the late regime. Late val (66-70) is a better selection metric; blend w=0.5 optimal on both (late 0.13542).
- Feature drift PSI train->test: activity/count features drift hardest (ord_nnew 0.130, ord_ncan 0.126, mk_cnt_total 0.125, tx_n 0.095).
- Queue: E3 log1p count features (running, /tmp/e3.log), E2 recency ramp re-scored on late val (eval_recency2.py), then cohort GBM, 1D-CNN on bar series.

## Sept 14 00:00-05:00 — Round 3: systematic pivot attempts (all dead ends except pseudo-sims)
- Seq GRU (60x6 bar seqs, 150K-200K subset): NEGATIVE signal (seedavg -0.017; blend worsens). Family discarded.
- Month-cohort GBM (train on 31/41-60): monotone worse (0.1236→0.1126). More data always wins.
- Drift-feature ablation (drop top-5 PSI): hurts late val (0.1266→0.1243). Drifting features carry real signal.
- Pseudo-labeling (self-training on forward months, w=0.1, confident half): POSITIVE in 2/3 forward sims (+0.0023 late on eval61-70, +0.0009 on eval66-70, neutral on 51-60). Applied to test: v2 LB 0.113 (=v1), v3 (GBM+MLP both pseudo) LB 0.112. LB-neutral at 3-decimal resolution.
- Adversarial importance weighting: AUC 0.74 (strong shift train-vs-forward) but weighted training hurts (0.1186 vs 0.1216). Dead.
- Ridge 3rd model: hurts blend at any weight. Dead.
- Blend weight sweep: flat 0.45-0.50. No lever.
- Diverse-config GBM (B_deep/C_shallow): blend +0.0002 full = noise. Dead.
- Big MLP (512-256-128): 0.1158 vs small 0.1251. Overfits. Dead.
- Rank-target GBM: solo 0.085, blend hurts. Dead.
- Fingerprint-view GBM: solo 0.040, blend hurts. Dead.
- 60-grid fingerprints (10s) for MLP: 0.1155 vs 30-grid 0.125. Noisier cells. Dead.
- LB probes (refs 56217171/56217183): blend 0.113 > GBM-only 0.110 > MLP-only 0.109. Diversity transfers; late-val ranking does NOT rank close variants on LB.
- Structural: month-70 blend cos 0.107 ≈ LB 0.113 → LB measures late regime. No temporal leak in test (sbp 0..597 all pre-predict).
- CHAMPION unchanged: GBM(58)+MLP(121) 50/50 unit blend, refit 0-70, LB 0.113, rank ~198.
- Box note: nan_to_num on huge arrays OOMs (3 full bool masks) — always blockwise. glibc heap creep needs malloc_trim; background jobs >1.2GB spike-killed, prefer foreground for builds.

## Sept 14 ~06:45 CEST — X10 (bucketed fino 6 cubetas + retorno por cubeta)
- features10.py: 6 cubetas sbp [0,30,90,180,300,450,600], 10 stats/cubeta (las 9 de X7 + ret first->last price) = 60 feats. Build train (1257637,60) 89s + test (647896,60) 48s. OOM inicial en fase derive -> fix in-place reuse de buffers (commit).
- gbm_x10.py (X2+X789+X10, 210 feats, tr<=60/va>=61, seeds 7/42, iters 104/103):
  solo full 0.125039 / late 0.128726 (X2+X7 ref 0.124663/0.127993).
  blend gbmX10+mlp 50/50: full 0.131743 / late 0.136141.
  blend X10/X789/mlp 25/25/50: full 0.132396 / late 0.136997 (~= v5 0.13237/0.13706, NO mejor).
- Lectura: X10 redundante encima de X789 (misma familia). Test limpio corriendo: X2+X10 vs X2+X7 (reemplazo) + shift evals (gbm_x10_ab.py).
- Disco: 100% -> borrados mlp60_Xstd (869M, dead end r3), x789 memmaps stale; 2.3G libres.

## Sept 14 ~06:50 CEST — X10 verdict: DEAD END
- x2x10 (X2+X10, 118f) main val: 0.124701 vs X2+X7 0.124663 (empate).
- x2x10 shift (tr<=50, va 51-60): 0.125640 vs X2+X7 0.125788 (peor).
- X789+X10 shift: 0.124458 vs X789 solo 0.12496 (anadir X10 EMPEORA shift -> overfit de cubetas finas).
- Conclusion: mas granularidad temporal NO ayuda; familia bucketed-by-sbp saturada con X7/X8/X9. No submission (regla 4).
- Siguiente: X11 familia estructural nueva (microestructura: Kyle lambda / Amihud por cubeta + time-to-event).

## Sept 14 ~07:00 CEST — X11 verdict: DEAD END
- x2x11 main val 0.123528 (vs X2 0.123587), x789x11 0.126161/0.130117 (vs X789 0.126590/0.131000 - peor), shift 0.123422 (vs X2 0.122894, ruido).
- Conclusion: ranges hi-lo / tv-range / sbp-span / time-to-event no aportan sobre X2 ni X789. Dos familias cerradas hoy (X10, X11).
- Siguiente: X12 = geografia de precios de ordenes (distancia al mid por cubeta, 2 pasadas) - lo no minado de order.feather.

## Sept 14 ~07:10 CEST — X12 POSITIVE (geografia de precios de ordenes)
- features12.py: distancia volume-weighted al mid de cubeta (2 pasadas: mid por (sid,bucket) de market, luego stream order) x (4 cubetas x new/can x buy/sell) = 32 feats. Build train+test OK.
- X2+X12: 0.127005 (vs X2 0.123587, +0.0034 - mayor salto que X7).
- X789+X12 (182f): 0.128181 full / late (vs X789 0.126590, +0.0016 incremental).
- Shift X2+X12: 0.124715 vs X2 0.122894 (+0.0018, positivo). Shift X789+X12 corriendo (ref X789 0.12496).
- BLEND gbmX789X12+mlp 50/50: full 0.133496 / late 0.138168 (v5: 0.132370/0.137060, +0.0011 ambos). w=0.55 late 0.138209.
- PLAN v6: si shift champion-config positivo -> refit 0-70 3 seeds @1.1x, blend 50/50 MLP, zero no-data, clip, submission tras reset cuota ~02:00 CEST 15 sept.

## Sept 14 ~07:35 CEST — v6 STAGED (esperando reset cuota ~02:00 CEST 15 sept)
- x789x12shift: 0.127664 vs X789 shift 0.12496 (+0.0027) — EVIDENCIA COMPLETA positiva.
- Refit X2+X7+X8+X9+X12 (182f) meses 0-70, seeds 7/42/123 @1.1x (109/97/102 it): modelos x7812_seed*.txt, refit_x7812_{test,train}.npy.
- submission_v6x7812.csv: 50/50 blend con refit_mlp, nodata=0, clip 0.1/99.9. In-sample train cos 0.203819.
- OOM fix en refit_x7812.py: test mode ya no construye la matriz train.
- Siguiente: X13 (trade geography + rms dist de ordenes) features13.py commiteado, build pendiente.

## Sept 14 ~07:50 CEST — X13 POSITIVE (trade geography + rms dist ordenes)
- features13.py: (a) trade price vs bucket mid por (cubeta,side): dist + absdist vol-weighted = 16; (b) rms dist de ordenes por (cubeta,new/can,side) = 16. Total 32.
- X2+X13: 0.127720 (mejor que X2+X12 0.127005).
- Champion (X789+X12+X13, 214f): val 0.129131 (+0.0010 vs X789X12), SHIFT 0.128812 (+0.0011 vs 0.127664) — doble positivo.
- Blend +MLP 50/50: full 0.134095 / late 0.138456 (nuevo mejor; v5 0.132370/0.137060).
- Refit 214f en cadena (trainmodels->test->trainpred) corriendo; submission v6 se rebuena con X13 al terminar. Cuota resetea ~02:00 CEST.
- Nota: X2+X12+X13 shift bajaba (0.123621) pero en config completa el shift SUBE — la interaccion con X789 importa.

## Sept 14 ~07:57 CEST — v6 FINAL staged (X12+X13)
- Refit 214f OK (memmap fix): x7813_seed{7,42,123}.txt, refit_x7813_{test,train}.npy.
- submission_v6x7813.csv (blend 50/50 MLP, nodata=0, clip 0.1/99.9, in-sample 0.210435). v6x7812 descartada.
- PENDIENTE: subir tras reset cuota ~02:00 CEST 15 sept via submit3.py. Reportar LB+rank.

## Sept 14 ~08:20 CEST — X14 verdict: DEAD END (redundante)
- xall14 (246f) val 0.128344 < X789X12X13 0.129131; shift 0.127747 < 0.128812. Bandas de distancia ya cubiertas por dist/rms de X12/X13. No incluir.
- X15 en build: level spacing del libro (askgap/bidgap/L1 vol share/L1-L2 volgap por cubeta).

## Sept 14 ~08:31 CEST — X15 verdict: FUERA (val+, shift-)
- xall15 (230f) val 0.129908 (+0.0008) PERO shift 0.127971 < 0.128812 (-0.0008). Regla 4: sin evidencia forward-sim fuerte -> NO entra en v6.
- v6 se queda con 214f (X789+X12+X13): submission_v6x7813.csv staged. Subir tras reset ~02:00 CEST.
- Familias cerradas hoy: X10 (neutro), X11 (neutro), X12 (+), X13 (+), X14 (redundante), X15 (mixto).
- Ideas agotadas en per-sample obvias. Proximas opciones: (a) geography por bandas de volumen (dist solo de ordenes grandes), (b) ensemble diversity: MLP con X12/X13 anadidas a sus 121 feats, (c) 3er modelo diverso (extra trees / linear sobre ranks) para blend de 3 vias.

## Sept 14 ~08:37 CEST — MLP+X12X13 dead end; X16 en build
- MLP con X12/X13 (185f): solo 0.126175/0.130731 (muy ruidoso: seed0 0.1198, seed1 0.1265). Blend gbm+newMLP late 0.137752 < 0.138456. NO mejora. v6 sigue: 214f GBM + MLP original.
- X16 (whale geography: dist/absdist separado por orden grande vs pequena vs media) construyendo.

## Sept 14 ~09:25 CEST — X16 (whale geography) fuerte en val, shift neutro
- X2+X16: 0.128829 (mejor adicion single). xall16 (246f): 0.132116 (+0.0030). Blend late 0.140090 (+0.0016 vs v6).
- Shift 51-60: 0.128442 vs 0.128812 (-0.0004, nivel ruido). Decidiendo con 2a ventana forward (train<=55, val 56-65) A/B 214f vs 246f corriendo.

## Sept 14 ~09:35 CEST — X16 IN. v7 (246f) refit corriendo
- Evidencia X16: val +0.0030 (0.132116), blend late +0.0016 (0.140090), X2+X16 shift +0.0037, midwindow (tr55/va56-65) 246f 0.131409 vs 214f 0.130587 (+0.0008), full-config shift 51-60 -0.0004 (ruido).
- refit_x7816.py: 246f, seeds 7/42/123 @1.1x (122/156/139). -> submission_v7x7816.csv reemplaza v6 al terminar.
- Limpieza disco: borradas matrices X10/X11/X14/X15 (dead ends) y .f32 viejos. 1.6G libres.

## Sept 14 ~09:40 CEST — v7 STAGED (246f con X16)
- submission_v7x7816.csv: refit 246f + MLP 50/50, in-sample 0.221116. Reemplaza v6 (v6x7813.csv conservada como fallback).
- A las ~02:00 CEST (reset cuota): subir v7 primero. Si LB v7 >= LB v5, considerar v6 como 2a sonda.
- X17 (whale trade geography) en build.

## Sept 14 ~09:50 CEST — X17 verdict: DEAD END
- X2+X17 0.123613 (plano), xall17 0.132169 (+0.00005), shift 0.127842 (-0.0006). Trades ejecutan en el touch: dist de trades es casi degenerada, split por tamano no aporta. Fuera.
- Familia geography minada: dentro X12 (orders dist), X13 (trades dist + orders rms), X16 (orders whale split); fuera X14 (bandas), X17 (trades whale).
- Siguiente: X18 = imbalance de COLOCACION (new-can neto) en zona near-mid vs far por cubeta.

## Sept 14 ~10:00 CEST — X18 verdict: DEAD END
- x2x18 0.124667, xall18 0.131526 (< 0.132116), shift 0.128342 (plano). Net placement por zonas no aporta sobre X12/X13/X16. Familia geography CERRADA.
- Siguiente: diversidad de modelo — MLP puro sobre las 246 feats del GBM (sin fp views) para blend de 3 vias.

## Sept 14 ~10:25 CEST — MLP246 (puro 246f) suma en blend 3 vias
- MLP246 solo: 0.128149/0.132194 (seed0 0.1283, seed1 0.1216).
- Blend 3way GBM246 40% + oldMLP 30% + MLP246 30%: 0.136630/0.140778 — MEJOR que v7 (0.135883/0.140090) en ambos.
- Refit MLP246 meses 0-70 corriendo (epochs 10/4/7 = 1.1x best). -> v8 3-way. Plan submission 02:00 CEST: v7 (2way, evidencia completa) y v8 (3way) como sondas.
- Fix memoria: nan_to_num/clip blockwise (full-matrix OOM a 1.6GB anon).

## Sept 14 ~10:47 CEST — v8 STAGED (3-way)
- Refit MLP246 OK (train cos 0.156/0.134/0.146 por seed). refit_mlp246_{train,test}.npy.
- submission_v8_3way.csv = 0.4*GBM246 + 0.3*MLP + 0.3*MLP246, nodata=0, clip. In-sample 0.207896.
- Plan 02:00 CEST: submit v7 (2way) primero, v8 (3way) segundo. Comparar LB para decidir la linea.

## 2026-09-14 ~15:00 — v7 reconstruida con GBM246 5-seed
- Refit 5 seeds (7/42/123/2024/31337, iters 1.1x = [122,156,139,129,134]) completo tras tormenta de VM forks/OOM ~13:10-14:10; script idempotente con skip-if-exists (refit_x7816c.py).
- submission_v7x7816.csv reconstruido: 0.5*unit(GBM246 5seed) + 0.5*unit(MLP refit), clip q0.1/99.9, 17 filas no-data a 0. Train cos in-sample 0.220339.
- Plan intacto: 02:00 CEST submit v7 (ahora 5-seed) primero, v8 segundo.

## 2026-09-14 ~15:05 — X19 (fill-ratio) en evaluación
- Nueva familia: ratio ejecución/colocación por (sid, sbp-bucket, side) — linked order+transaction streams; GBM no puede dividir entre streams. 32 feats (fill, fillnet, exec_logratio, cancelover x 4 buckets x 2 sides).
- Hipótesis: presión de demanda vs provision de liquidez por zona temporal. Temporal-bucket ya cubierto por X7/X8/X9 (X19 no duplica: es cross-stream ratio).
- Eval corriendo: x2x19, xall19 (278f) val + shift vs refs 0.132116/0.128442 (gbm_x19_ab.py).

## 2026-09-14 ~21:30 — X19 (fill-ratio) MUERTA
- x2x19 solo 0.123059 < X2 0.123587; xall19 (278f) 0.131503 < 246f 0.132116; shift 0.128174 < 0.128442. Las 3 pruebas negativas. Familia fill-ratio cerrada.
- Matrices X19 borradas. Estado de familias: geography (X12/X13/X16) IN, todo lo demás probado muerto: X10/11/14/15/17/18/19.
- Siguiente hito: 02:00 CEST submit v7 (5-seed) + v8 probe.

## 2026-09-15 02:10 CEST — LB v7 y v8
- v7 (ref 56241229, GBM246 5-seed + MLP 50/50): LB 0.116 (vs v5 0.114). Nuevo mejor LB.
- v8 (ref 56241246, 3-way 0.4/0.3/0.3): LB 0.116 — empate con v7 a resolución del LB.
- LB val->LB: blend val 0.1359-0.1366 se traduce en +0.002 LB. LB sigue comprimido en 3 decimales.
- Cuota restante hoy: 3/5. Rank actual: 197/214 con 0.116 (mismo rank que ayer, score mayor; leaderboard CSV via API, 214 equipos).

## 2026-09-15 02:18 — probe estructural: y rank-transform por mes
- Hipótesis: entrenar con y = rank pct por mes ([-1,1]) en vez de y raw mejora generalización cross-régimen (métrica cos es invariante a escala global pero no a mezcla de regímenes).
- Eval: gbm_xrank_ab.py, 246f, val full + shift, refs rawY 0.132116 / shift 0.128442. Preds evaluados contra y RAW (la métrica real).
- Leaderboard screenshot rank 197 entregado a parent 02:17. Cuota 3/5 intacta.

## 2026-09-15 04:10 — rank-y MUERTO
- Rank-y (per-month pct rank): val 0.088312 vs 0.132116 raw; shift 0.097293 vs 0.128442. best_iter=1 en ambos — early stopping por L2 no sirve con targets rankeados y la info cross-month de escala (std por mes varía 2.6x) sí carga señal para cos. Descartado.
- corr(rank_y, raw_y)=0.86; month std range 0.0019-0.0050. Dato útil: la métrica cos premia acertar magnitudes de meses volátiles.

## 2026-09-15 15:38 — cos-ES GBM verdict: TIE, closed
- xallcos (GBM 246f, early-stop on cos feval, idea from public LGBM baseline): val 0.132069 vs L2-ES ref 0.132116; shift 0.128442 = ref exactly.
- No gain on either axis -> NOT a v9 component. Cycle logged per Diego's loop (search->check->optimize->ceiling->restart).
- Next: MLP v2 (cos-loss+EMA) verdict pending; then X20 (window-deltas + cross-stream interactions, UnseenAnchor v3-style, gained on all their folds).

## 2026-09-16 00:56 — MLP v2 (cos-loss + EMA 0.999) verdict: NEGATIVE, closed
- New MLP solo seed0 0.12623 (old MLP246 solo 0.128149). Blend gbm+newMLP 0.135729/0.138873 vs gbm+oldMLP 0.135883/0.140090. 3-way 0.136033/0.140061 vs ref 0.136630/0.140778.
- cos-loss+EMA loses on both val and shift -> keep old MLP. Idea from public RealMLP kernel does not transfer to our MLP246.
