# Screen X32 — extensión ctx seed 42 + ensemble 2 seeds (30-Sep-2026) — X32 CERRADO

Runs: mscapital-screen-x32-gpu v2 (seed 2026, 3.25h) + v3 (seed 42, 1.98h). Receta idéntica (champion 450f), folds 0,4, origin 70. Métrica: coseno no centrado (oficial). Scores single-seed recomputados desde predictions.npz coinciden con done.json.

## Deltas individuales ctx − flow31 (mismo fold/seed)
| fold | seed 2026 | seed 42 |
|------|-----------|---------|
| 0 | +0.000776 | −0.002200 |
| 4 | +0.000654 | +0.000955 |

## Ensemble 2 seeds (media de preds unit-normalizadas)
| fold | flow31 ens | ctx ens | delta |
|------|-----------|---------|-------|
| 0 (m40-44, n=88835) | 0.147343 | 0.146638 | −0.000705 |
| 4 (m65-70, n=104770) | 0.175671 | 0.176532 | +0.000861 |

## Desglose mensual (delta ctx ens − flow31 ens)
- fold 0: m40 −0.000214, m41 −0.002035, m42 −0.002307, m43 −0.000873, m44 +0.001742 (4/5 negativos)
- fold 4: m65 −0.001172, m66 +0.001314, m67 +0.002228, m68 −0.001048, m69 +0.002623, m70 −0.000873 (3/6 positivos)

## Veredicto
Puerta +0.002 estable: NO superada. Signos mixtos entre seeds (f0: +2026 / −42), ensemble mixto (−f0 / +f4), mensual mixto. El ruido de seed (~0.003, documentado) supera la señal ctx (~+0.0007 en seed 2026).
Decisión de Diego (WhatsApp 30-Sep 16:52:21): "Si no supera la puerta con consistencia, cierra X32 y conserva el campeón 0.141." → **X32 CERRADO. Campeón 0.141 (submission #1, ref 56634145) se conserva.**

Cuota GPU consumida en X32: v1 41s + v2 3.25h + v3 1.98h ≈ 5.2h. Restante ~10.8h (refresh 3-Oct).
Análisis completo: /tmp/screen_v3/ensemble_analysis.json.
