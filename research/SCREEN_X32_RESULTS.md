# Screen X32 — resultados (GPU v2, 30-Sep-2026)

Run: mscapital-screen-x32-gpu v2, 08:07→11:46 CEST (3.25h GPU, 6/6 jobs done, seed 2026, origin 70).
Métrica: coseno no centrado (oficial). es_selected_score = score de selección early-stopping en ventana ES del fold.

| fold | flow31 (control) | flow31ctx | flow31per | ctx−ctl | per−ctl |
|------|-----------------|-----------|-----------|---------|---------|
| 0 (ES m40-44) | 0.145155 | 0.145931 | 0.146473 | +0.000776 | +0.001318 |
| 4 (ES m65-70) | 0.174990 | 0.175645 | 0.174892 | +0.000654 | −0.000098 |

Media ctx−ctl: +0.000715 (2/2 folds positivo, consistente)
Media per−ctl: +0.000610 (signo mixto, ruido)

Puerta de promoción (era audit): +0.002 estable vs control, misma métrica/protocolo, ambos bloques, varias seeds.
Veredicto screen: NINGÚN arm cruza la puerta. ctx = señal consistente pero ~1/3 de la puerta; per = sin señal.

Artefactos: 6x done.json/history.json/predictions.npz/best.pt en el kernel (run/jobs/*). Hashes en done.json. Log sesión: /tmp/gpu_session_log.txt (local).
Cuota GPU restante: ~12.8h (refresh 3-Oct).
