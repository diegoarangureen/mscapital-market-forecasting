# MSCapital community research (2026-09-15)

Fuentes: discussion board oficial + notebooks públicos (material público, permitido; datos/modelos externos PROHIBIDOS por el host).

## Reglas/entorno
- Host: prohibido cualquier dato o modelo externo; sí vale procesar los datos del organizador (discussion/739945).
- Datos probablemente A-share L2 chino (streams order+transaction+snapshots 3s). Literatura de microestructura A-share aplica.
- Zigui Wang (19º): con val local, techo del public LB ~0.16; seed-lottery sobre public LB existe pero muere en private.

## bestwater (24º) — discussion/733271 (minado)
- CV↔LB inconsistente para todos. Transformer CV 0.155 → LB 0.120 (sobreajuste CV). LGBM 0.141 CV → 0.119.
- Su protocolo: 5-fold TS purgado (gap 2 meses), target rank-normalizado por mes a [-1,1], métrica cos sobre y normalizada.
- Sus mejores single-models (public LB): TabM 0.142, RealMLP/PTK 0.137, ResMLP 0.139, LGBM 0.131.
- **holdout+early-stop > refit-on-full**: TabM 0.142→0.137, ResMLP 0.129→0.100 al refitear 100% datos con épocas fijas. (Nuestro refit usa 1.1x iters, no fijas; sin A/B propio en LB.)
- Blend diverso (TabM+RealMLP+ResMLP+CNN-Transformer+externos) 0.147-0.148. La diversidad de ARQUITECTURA es su palanca.
- Distribución-alignment: CV 0.172 → LB 0.120 (trampa).

## Notebooks públicos (copiados a research/kernels/)
- sweetyheehee/lightgbm-baseline-salute-rib: features = nuestros X2/X7/X8/X9 (cubierto). CLAVE: `feval=cos_uncenter` para early stopping de LGBM (nosotros ES por L2). Params: lr 0.025, leaves 48, l2 10, max_bin 63 (más regularizado que el nuestro).
- yunsuxiaozi/rfmf-realmlp: RealMLP completo — PBLD embedding (features periódicas cos), NTPLinear ensemble, loss = MSE + 0.01*cos + 0.1*RQ-KMeans aux, EMA de pesos (0.999), flat-anneal LR, train_bs 256, epochs 10, feature-selection por correlación.
- yunsuxiaozi/rfmf-0726data + rfmf-lightgbm: mismas features.
- yangq369/submit-lb142: wrapper sobre dataset privado (pesos subidos), código no visible.
- Youler (8º): "EMA sobre datos completos mejora un poco" = EMA de pesos del modelo.

## Acciones adoptadas (en orden)
1. GBM: feval cos para early stopping (eval xall16 con feval cos vs 0.132116 ref).
2. MLP v2: cos-loss + EMA de pesos (vs MLP actual).
3. Si 1+2 mejoran: blend candidato a submission con evidencia.
4. RealMLP/TabM port completo = apuesta estructural grande (evaluar viabilidad CPU 2GB).
- Rank-y ya muerto en GBM propio (0.088 vs 0.132); bestwater lo usa pero con protocolo purgado + NNs. No reabrir por ahora.
