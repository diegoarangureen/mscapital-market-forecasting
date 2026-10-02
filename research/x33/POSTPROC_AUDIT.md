# X33 paso 1 y 3: auditoría CPU del post-procesado (2-Oct-2026)

Solo CPU, predicciones ya existentes (kernel mscapital-confirm-tpu v1, 5 folds, flow31 vs base). Script: `postproc_audit.py`, datos: `postproc_audit.json`.
Cuatro variantes fijadas antes de mirar: V0 raw / V1 solo clip / V2 solo ceros no-data / V3 ambos (= actual). Sin búsqueda de umbrales.

## Global (coseno no centrado)
| Arm | V0 raw | V1 clip | V2 ceros | V3 ambos |
|---|---|---|---|---|
| flow31 | 0.165621 | 0.160042 | 0.165621 | 0.160042 |
| base | 0.162929 | 0.157228 | 0.162929 | 0.157228 |

Sin mes 66: flow31 raw 0.149330 vs procesado 0.145967; base 0.146904 vs 0.143698. El mes 66 aporta mucho pero el clip pierde en todo caso.
- La pérdida (-0.0056) viene entera del clip. Los ceros no hacen nada aquí (filas no-data en score = 0, share de Σy² = 0).
- Delta pareado flow31 - base: raw +0.00269, clip +0.00281. La comparación entre ramas no depende de la variante.

## Por origen (raw -> clip), flow31 / base
- origen 59: 0.17976 -> 0.17395 / 0.17749 -> 0.17161
- origen 64: 0.14911 -> 0.14399 / 0.14564 -> 0.14042

## Por seed (pérdida raw - clip)
- flow31: o59 2026 0.00583, 42 0.00583; o64 2026 0.00547, 42 0.00482
- base: o59 2026 0.00557, 42 0.00634; o64 2026 0.00487, 42 0.00549
Consistente en seeds y orígenes (siempre ~ -0.005).

## Por mes (pérdida raw - clip), flow31
m62 0.0036, m63 -0.0029, m64 0.0024, m65 -0.0028, m66 0.0079, m67 0.0061, m68 0.0099, m69 0.0046, m70 -0.0021
(base similar). Pierde en 6 de 9 meses; los meses 66-68 concentran la pérdida. Delta pareado flow31-base por mes (raw): m62 +0.0025, m63 -0.0004, m64 +0.0007, m65 -0.0010, m66 +0.0039, m67 +0.0065, m68 +0.0025, m69 +0.0017, m70 +0.0027.

## Los límites de clip por fold no encajan con el ensemble completo
Los límites son q0.001/q0.999 de la media ES, y en ES cada fila tiene solo 2 modelos (las 2 seeds de un fold); las filas de score tienen 10 modelos. Std raw del ensemble de score 0.0545/0.0561 vs std ES 0.0467/0.0455. Los límites se ajustaron a otra distribución.
| Origen | Límites | q0.001 raw | q0.999 raw | clip bajo | clip alto |
|---|---|---|---|---|---|
| 59 | [-0.2391, +0.2748] | -0.3263 | +0.3550 | 0.26% | 0.25% |
| 64 | [-0.2310, +0.2477] | -0.3033 | +0.3120 | 0.25% | 0.27% |
Se recorta ~0.5% de filas, justo las de señal extrema.

## Contribución de la cola (flow31 raw, ensemble)
- Top 1% |y|: 29.2% de Σy² y 29.2% de Σp·y.
- Top 1% |p|: 28.6% de Σp², 33.3% de Σp·y, 8.5% de Σy².
- Coseno sin el top 1% |y|: 0.1455 (vs 0.1656).
- 37.6% de las filas top-1% |y| tienen p·y < 0.
- Diagnóstico: cos(p, y winsorizado q0.5/q99.5) = 0.1608 (no es una métrica de selección).

## Riesgo de selección declarado
Estos bloques ya se consultaron para comparar flow31 vs base. Cualquier regla de post-procesado elegida mirando estos números queda seleccionada, y no la valida. Además el campeón LB 0.141 usó el clip de origen 70 ajustado con las predicciones del ensemble final (otra distribución a su vez). Este informe explica la brecha; no propone un umbral ni cambia nada. Cualquier cambio de post-procesado necesita decisión explícita de Diego y confirmación en datos no consultados.

---
# Paso 2: informe X32 recalculado con la receta campeón (media directa, X32 sigue cerrado)
`x32_direct_mean.py` / `.json`. Predicciones de v2 (seed 2026) y v3 (seed 42).
| fold | flow31 | ctx | delta directa | delta unit-norm anterior |
|---|---|---|---|---|
| 0 | 0.147374 | 0.146637 | -0.000737 | -0.000705 |
| 4 | 0.175681 | 0.176528 | +0.000846 | +0.000861 |
Singles sin cambio (f0 +0.000776/-0.002200; f4 +0.000654/+0.000955). Meses directa f0: m40 -0.00017, m41 -0.0021, m42 -0.00236, m43 -0.00092, m44 +0.00168; f4: m65 -0.00109, m66 +0.00123, m67 +0.00222, m68 -0.00108, m69 +0.00263, m70 -0.00075.
La conclusión no cambia: puerta +0.002 no superada, X32 cerrado, campeón 0.141.
