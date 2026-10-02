# X33 screen (2-Oct-2026): Control / A1 / A2, folds 0 y 4, seeds 2026/42 - DETENIDO por Diego (20:26)

Receta campeón, un solo cambio por brazo. Métrica: coseno no centrado sobre el target original. Código: `train_audited.py` (A1 = `target_winsor`), configs `configs/audited/x33_*.json`, análisis `screen_analysis.py` / `.json`.

## Control (fresco): reproduce X32 bit a bit
f0 s2026 0.145155, f0 s42 0.147578, f4 s2026 0.174990, f4 s42 0.173981. Coinciden con el flow31 de X32. El cambio de trainer no altera el comportamiento por defecto.

## A1: winsorize target de train q0.5/q99.5 (bounds por fold solo con train, guardados en target_winsor.json; clip antes del ruido; ES y eval con target original)
| fold | s2026 | s42 | ensemble (media directa) |
|---|---|---|---|
| 0 | +0.0007 | -0.0006 | -0.00006 |
| 4 | -0.0075 | -0.0056 | -0.0068 |
Meses f0 positivos 1/5, f4 2/6 (m66-m69: -0.005 a -0.008). **No pasa la puerta (+0.002 estable).** Recortar el target quita señal de cola que la métrica premia (top 1% |y| = 29% de Σy²).

## A2: angular_loss cosine - INCOMPLETO (cancelado)
Solo 2 de 4 modelos: f0 s2026 0.145320 (delta +0.000165), f4 s2026 0.174083 (delta -0.000907). Faltan s42 en ambos folds. Sin veredicto formal; los dos deltas disponibles están muy por debajo de +0.002.

## GPU
Screen consumió ~20.5k s (~5.7h) de cuota GPU (69169 -> 89712 s), por debajo del tope de 6.0h. Pérdida: control y A1 pararon al tope de 2.0h con el 4º modelo en epoch 8/10 (rescatado con resume, ~0.6h cada uno); A2 se canceló a mitad de su 3er modelo.

## Estado
X33 detenido por Diego antes de completar A2. Ningún submission. Campeón 0.141 intacto.
