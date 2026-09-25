# LOCAL TRAINING HANDOFF — SUPERSEDED, READ TRAINING_AGENT.md FIRST

**Sep 25 14:25: Diego's audit (commit e0ce641) replaced this pipeline.**
The canonical guide is now [TRAINING_AGENT.md](TRAINING_AGENT.md): new
entrypoint (`src/kaggle/train_audited.py --config ...`), corrected feature
builders (X30v2/X31v2 — the old X30/X31 packs had an inverted-chronology bug
and must not be used for training), fixed OOF/metric handling, and a new
promotion gate (paired +0.002 vs control on the same metric/protocol — the
old absolute `val >= 0.130` gate is retired). Official metric verified
2026-09-25: uncentered `cos(prediction, target)` per the competition's
Evaluation tab. Sections below are kept ONLY for the data-download commands
and the historical model description; every training instruction in them is
stale. Budget: 30h GPU + 20h TPU (Kaggle), ceilings not targets.

---

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
