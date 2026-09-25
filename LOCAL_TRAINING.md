# Entrenamiento local: pipeline auditado

La gu?a vigente es [TRAINING_AGENT.md](TRAINING_AGENT.md). Sustituye el handoff
anterior y contiene comandos de construcci?n, preparaci?n, entrenamiento,
confirmaci?n temporal, reanudaci?n, empaquetado Kaggle y exportaci?n.

- Entrenador: `src/kaggle/train_audited.py`, CPU/CUDA/XLA expl?citos.
- Configuraciones: `configs/audited/`.
- Nuevos packs: **X30v2** (36 columnas) y **X31v2** (18 columnas); reconstruir
  train y test. Base: 450 columnas; base + FLOW: 468; base + X30v2: 486;
  base + FLOW + X31v2: 486.
- Pruebas: `python -m pytest -q`.
- Verificar el scorer oficial antes del modo final. Se registran coseno y Pearson.
- No comparar los scores nuevos directamente con el antiguo `cos_np`, que
  centraba ambos vectores. El OOF hist?rico agregado tambi?n sobrescrib?a seeds.
- `rescore_legacy_oof.py` permite recuperar el promedio por seed sin reentrenar.
- Los archivos originales se conservan bajo `src/kaggle/legacy/` para reproducir
  el historial; no usarlos para nuevos experimentos.

[Handoff anterior, archivado](research/LOCAL_TRAINING_PRE_AUDIT_20260925.md).
