# Campeón congelado: flow31 v1 (LB 0.141) — 2026-09-28

Congelado según el paso 1 de la decisión de Diego (WhatsApp 28-sep 11:17):
configuración, features, checkpoints, predicciones y submission. Este es el
resultado conservado; todo experimento posterior se mide contra esta receta
exacta.

## Configuración
- `configs/audited/final.json` (commit 582acb2): mode final, arm `flow31`,
  seeds {2026, 42, 123}, folds {0..4}, origen 70, weight_target noisy,
  métrica coseno oficial (verificada 2026-09-25, fuente en el propio config).
- Receta RealMLP: BS 256, 10 epochs flat-anneal, patience 3, EMA 0.998,
  MSE pesado (w=0.5 |y|>0.001), label noise 0.005 anneal, scaler robusto
  mediana/IQR soft-clamp ajustado por fold (preservada por la auditoría).

## Features (486)
- base 450f: 298 propietarias + 152 públicas (builders en `src/`, matriz
  `base_{train,test}.npy` del dataset preparado).
- FLOW (X30v2, 18f del grupo flow): `src/kaggle/build_x30.py` (cronología
  corregida post-auditoría).
- X31v2 (18f): `src/kaggle/build_x31.py` (cronología corregida).
- Dataset preparado: https://www.kaggle.com/datasets/diegoaranguren/mscapital-prepared-v2
  (manifest con sha256 por archivo; `src/kaggle/audit/dataset.py` verifica
  firma y checksums en carga).

## Checkpoints
- Kernel Kaggle `mscapital-final-gpu` v1 (salida persistente): 15 jobs
  `flow31_o70_f{0..4}_s{2026,42,123}`, cada uno con `best.pt`,
  `predictions.npz` (OOF + test del modelo) y `history.json`.
- La copia local /tmp es efímera; la salida del kernel es el almacén canónico.

## Predicciones
- `aggregate_flow31.npz` (en la salida del kernel final): OOF ventana de
  ajuste 459,901 filas (raw/processed) + test 647,896 (raw/processed).
- Predicciones externas independientes para ensembles futuros: salida del
  kernel `mscapital-confirm-tpu` v1 (`aggregate_base.npz`,
  `aggregate_flow31.npz` sobre orígenes 59/64, bloques 62-66 y 67-70).

## Submission
- `research/submission_flow31_v1.csv` (sha256 c2e5e71c03bcc75c,
  647,896 filas). Ref Kaggle 56634145, estado COMPLETE.
- LB público: **0.141** (rank ~125/253, 28-sep). Anterior mejor: 0.139.

## Medida independiente (confirm §5)
- summary archivado: `research/confirm_flow31_2026-09-27.json`.
- flow31−base = +0.002814 global; +0.002342 (62-66), +0.003567 (67-70);
  ambas seeds positivas; LOMO 9/9; bootstrap 95% [+0.00104, +0.00388].
- Calibración CV→LB observada: +0.0028 CV → +0.002 LB (una sola submission;
  NO usar como factor de descuento sistemático — matiz de Diego 28-sep).
