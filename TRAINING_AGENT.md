# Handoff al agente de entrenamiento — auditoría del 25-09-2026

## Mensaje de Diego y objetivo

Usa el pipeline auditado de este repositorio. Conserva v13 (LB 0,139) como
referencia histórica y evalúa un nuevo control reproducible de 450 features.
Presupuesto total disponible indicado por Diego: **30 h GPU y 20 h TPU**.
No se ha ejecutado entrenamiento competitivo durante esta auditoría.
No asumas una mejora de leaderboard: los cambios corrigen construcción,
medición y reproducibilidad; su efecto predictivo debe medirse.

**Entrypoint canónico:** `src/kaggle/train_audited.py`.
Los nueve entrypoints activos anteriores redirigen a él y requieren `--config`.
Sus implementaciones originales están en `src/kaggle/legacy/` exclusivamente
para reproducir el historial. Los demás scripts antiguos del repositorio son
experimentos históricos; no son una vía alternativa para nuevos entrenamientos.

## Qué cambió y qué no debes mezclar

- `audit/metrics.py` centraliza coseno sin centrar y Pearson. Se registran ambos;
  `metric` elige early stopping, comparación y score después del posprocesado.
- Los JSON suministrados usan **coseno provisional**. No hemos podido obtener
  el scorer ejecutable oficial. Verifica su definición en la competición,
  conserva código/fuente y fecha, y completa `metric`, `metric_verified: true`
  y `metric_source`. Si resulta ser Pearson, cambia TODAS las configuraciones
  antes del entrenamiento definitivo. No marques verificado por intuición ni
  por coincidencia aproximada con un score público. `final` exige esa evidencia;
  screen/confirm permiten diagnóstico provisional con ambas métricas.
- OOF: suma y contador por fila/variante; nunca sobrescribir una seed ni mezclar
  brazos. Un resultado de early stopping NO es evaluación externa.
- `X30v2` y `X31v2`: tiempo cronológico descendente en
  `seconds_before_predict`; samples completos entre chunks/batches IPC; empate
  temporal conserva orden original; OFI omite transiciones con libro vacío;
  autocorrelación de signos usa número de pares válidos; buckets [0,10),
  [10,30), [30,60); denominador Kyle usa exactamente pares válidos;
  gaps NEW se calculan después de filtrar NEW; referencia mid por sample,
  sin percentiles globales ajustados por separado en train/test.
- Rehacer **train y test**. Los nuevos builders escriben `X30v2_*`, `X31v2_*`,
  IDs y manifiestos; no sobreescriben los packs X30/X31 históricos.
- Arquitectura, scaler robusto train-only, EMA, batch 256 y optimizador conservan
  la base del campeón. El schedule por época es idéntico en CPU/CUDA/XLA.
  No es una réplica numérica del antiguo schedule GPU por paso.
- El peso del MSE puede depender de target limpio (`clean`), ruidoso (`noisy`)
  o ser uniforme. Los controles y cribados de features fijan explícitamente
  `noisy` para aislar el cambio de features. `clean_weights.json` mide solo esa
  corrección frente a `noisy_weights_control.json`.
- RNG XLA explícito. Permutaciones y ruido usan generadores CPU separados de
  la inicialización, por modelo/época. El orden de ejecución no debe alterar
  un fold. Una misma seed no significa pesos idénticos entre arquitecturas.
- Se conservan `last.pt` por época (modelo, EMA, optimizador, RNG, scaler,
  paciencia e historial), `best.pt`, predicciones e IDs por modelo.
  `done.json` se escribe al final, con checksums. Después se elimina el `last.pt`
  del modelo completado para ahorrar disco (se conserva `best.pt` completo).
  Usa `keep_completed_last: true` si necesitas también sus estados de optimizador.
  Repetir el comando reanuda;
  no se pueden mezclar recetas, código, datos o backends en un mismo directorio.

## 1. Preparación del entorno

Trabaja desde la raíz del repositorio. Los comandos siguientes son para Linux/
Kaggle. En Windows puede usarse `.venv/Scripts/python.exe` en lugar de `python`.

```bash
python -m pip install -r requirements-audit.txt
python -m pytest -q
```

Instala por separado el build PyTorch apropiado para tu GPU local. En Kaggle TPU
conserva el par `torch`/`torch_xla` preinstalado compatible: no lo sustituyas con
un `pip install torch` genérico. La auditoría local se prueba con CPU; no certifica
rendimiento ni determinismo XLA/CUDA.

Descarga las fuentes autorizadas de la competición:

| Argumento | Fuente |
|---|---|
| `--data` | `diegoaranguren/mscapital-matrices` |
| `--x21` | `diegoaranguren/mscapital-x21` |
| `--x22` | `diegoaranguren/mscapital-x22` |
| `--public` | output `yunsuxiaozi/rfmf-0726data`, train.csv/test.csv |
| `--labels` | raw `train/label.feather` de la competición |
| builders `BASE` | carpeta con train/ y test/ de la competición |

La preparación exige 450 columnas finales, 152 públicas, IDs densos alineados,
y compara target/month de las matrices con el label.feather original. X21/X22
conservan el contrato histórico de builders indexados por sample_id; sus archivos
no contienen IDs propios. Los hashes congelan los artefactos consumidos.

## 2. Construir features en CPU y congelar dataset

Ejemplo; sustituye las rutas:

```bash
export BASE=/ruta/competition
export OUT=/ruta/features-audit
mkdir -p "$OUT"
SPLIT=train NS=1257637 python src/kaggle/build_x30.py
SPLIT=train NS=1257637 python src/kaggle/build_x31.py
SPLIT=test NS=647896 python src/kaggle/build_x30.py
SPLIT=test NS=647896 python src/kaggle/build_x31.py

python src/kaggle/prepare_audited.py \
  --data /ruta/matrices --x21 /ruta/x21 --x22 /ruta/x22 \
  --public /ruta/rfmf-0726data --labels "$BASE/train/label.feather" \
  --x30 "$OUT" --x31 "$OUT" --include-test --out prepared
```

Usa una carpeta `prepared` vacía. No cambies sus archivos tras publicar el
manifiesto. Los lectores rechazan checksums incorrectos y confusiones v1/v2.
Los feather comprimidos pueden descomprimir un record batch enorme: el nuevo
reblocking arregla continuidad, **no convierte Arrow en IO de memoria constante**.
Haz el build en una sesión CPU con RAM suficiente; comprueba el pico antes de
procesar todo (32 GB es una referencia operativa, no una medición nueva).
Si sample_id no está agrupado globalmente en orden ascendente, el builder aborta;
ordena/prepara la fuente explícitamente, no desactives la comprobación.

## 3. Piloto de control y reproducibilidad

```bash
python src/kaggle/train_audited.py --config configs/audited/replication.json --plan
python src/kaggle/train_audited.py --config configs/audited/replication.json \
  --data prepared --out runs/repl_gpu_a --device cuda --max-hours 1
python src/kaggle/train_audited.py --config configs/audited/replication.json \
  --data prepared --out runs/repl_gpu_b --device cuda --max-hours 1
python src/kaggle/summarize_audited.py --data prepared --out runs/repl_gpu_a
```

Repite el piloto en TPU con `--device xla`, en directorios distintos. Compara
predicciones además de métricas. Ejecuta la misma clave `--job-key` aislada y
después de otros modelos en dos outputs; invierte el orden de brazos en un
config separado si quieres medir ese efecto. No alteres un config a mitad de
una ejecución. Una variación del orden de 0,002/0,003 debe investigarse, no
rebautizarse automáticamente como ruido de sesión.

Registra backend, versiones, chips/dispositivos utilizados, tiempo total,
compilación, tiempo por modelo y memoria. `run_worker_*.json` recoge parte de
estos datos. `--max-hours` se comprueba entre modelos/épocas: **es un límite
blando**, puede excederse en una época más guardado/inferencia. Deja margen
respecto del límite duro de Kaggle. Reanuda con el mismo comando/output.

## 4. Cribado de features y pérdida

```bash
python src/kaggle/train_audited.py --config configs/audited/screen.json --plan
python src/kaggle/train_audited.py --config configs/audited/screen.json \
  --data prepared --out runs/screen_tpu --device xla --max-hours 4
python src/kaggle/summarize_audited.py --data prepared --out runs/screen_tpu
```

`screen.json`: **12 modelos** = 4 brazos × 3 folds × 1 seed.
Base 450; X30v2 completo 486; FLOW 468; FLOW+X31v2 486.
Usa `flow31-flow` para medir el incremento X31. Este resultado usa las mismas
ventanas para elegir época y puntuar: solo sirve para cribado.

`family_interaction.json` es opcional y condicionado al presupuesto; compara
FLOW, PRICE, FLOW+PRICE y X30 completo, incluyendo aporte condicional de VOL.
No lo ejecutes automáticamente además de todo lo demás.

Para medir pesos limpios del MSE, ejecuta los dos configs siguientes en el mismo
backend, con outputs separados; son seis modelos por config:

```bash
python src/kaggle/train_audited.py --config configs/audited/noisy_weights_control.json \
  --data prepared --out runs/loss_noisy --device cuda --max-hours 3
python src/kaggle/train_audited.py --config configs/audited/clean_weights.json \
  --data prepared --out runs/loss_clean --device cuda --max-hours 3
python src/kaggle/summarize_audited.py --data prepared --out runs/loss_noisy
python src/kaggle/summarize_audited.py --data prepared --out runs/loss_clean
python src/kaggle/compare_audited.py --data prepared \
  --base runs/loss_noisy --candidate runs/loss_clean --out runs/loss_comparison.json
```

Inspecciona `history.json`: MSE, pérdida angular y normas de sus gradientes en
el primer batch de cada época. No abras un grid de lambdas sin esta medición.
No aumentes batch size ni cambies precision/EMA a la vez que las features.

## 5. Confirmación del ensemble completo

Congela el ganador del cribado en una copia de `confirm.json`; `x30` es un
placeholder, **no un ganador ya elegido**. Fija también la receta de pérdida.
El mismo config contiene control y candidato, salvo que estés comparando dos
pérdidas: en ese caso usa configs separados y `compare_audited.py`.

```bash
python src/kaggle/train_audited.py --config configs/audited/confirm.json --plan
python src/kaggle/train_audited.py --config configs/audited/confirm.json \
  --data prepared --out runs/confirm_tpu --device xla --max-hours 10
python src/kaggle/summarize_audited.py --data prepared --out runs/confirm_tpu
```

Son **40 modelos**: 2 variantes × 2 seeds × 5 cortes × 2 orígenes.
Al origen 59, desplaza toda la plantilla temporal 11 meses hacia atrás: ningún
entrenamiento ni early stopping ve labels posteriores a 59; puntúa 62–66.
Al origen 64 desplaza la plantilla seis meses: puntúa 67–70.
Cada fila externa recibe el promedio de los cinco cortes y todas las seeds,
como en test. Los límites de clipping se ajustan al promedio de predicciones
de validación INTERNA de cada origen, antes de ver su bloque externo.
Se reporta el coseno sobre el vector externo completo, nunca el promedio de
coseno por batch/fold. Mes 66 está incluido; excluirlo es sensibilidad secundaria.

`summary.json` contiene deltas pareadas, por origen, por seed, por mes,
leave-one-month-out e intervalo
bootstrap por mes. Con pocos meses/dependencia y selección histórica previa,
el intervalo es orientativo. Estos bloques ya se usaron en investigación anterior;
no los presentes como holdouts nunca vistos. Congela decisiones antes de abrir
el nuevo informe; consultar el bloque repetidamente vuelve a contaminar la selección.

Gate de promoción: mejora del orden de +0,002 estable respecto al control con
misma métrica/protocolo, evidencia en ambos bloques y varias seeds, sin depender
de un solo mes. Es una regla operativa, no un test estadístico automático.
Conserva todos los deltas, incluso negativos. No uses el antiguo gate absoluto
`val >= 0.130`, ni el score histórico Pearson como umbral del nuevo coseno.

Los resultados incompletos NO se comparan: `summary.json` lista jobs pendientes.
No mezcles modelos CUDA y XLA en un output para completarlo. Cada backend tiene
su control y su provenance. Una tercera seed es confirmación adicional si queda
cuota, no una obligación para gastar la reserva.

## 6. Entrenamiento final e inferencia

Solo después de escoger receta y verificar scorer, copia/edita `final.json`
con un único brazo ganador y tres seeds. Son 15 modelos. No cambies la receta
ganadora entre confirmación e inferencia ni hagas un refit nuevo sobre todo train.

```bash
python src/kaggle/train_audited.py --config configs/audited/final.json \
  --data prepared --out runs/final_gpu --device cuda --max-hours 9
python src/kaggle/summarize_audited.py --data prepared --out runs/final_gpu \
  --submission-template /ruta/competition/submission.csv
```

La exportación alinea por sample_id, exige cobertura total y finitud, y pone
nodata a cero DESPUÉS de clipping. Se guarda el CSV; no se envía a Kaggle.
Conserva v13 y el candidato con sus hashes. Informa a Diego antes de publicar
una submission; esta petición prepara el pipeline y el handoff, no ordena enviarla.

## 7. Kaggle sin checkout y uso opcional de varios chips

Para un kernel de un solo archivo:

```bash
python src/kaggle/bundle_audited.py --entry train_audited.py \
  --config configs/audited/screen.json --out dist/screen.py
python dist/screen.py --plan
```

Adjunta el dataset preparado y configura argumentos `--data /kaggle/input/...`,
`--out /kaggle/working/run`, `--device xla`. También puedes empaquetar los builders,
preparación, rescore y resumen con `--entry` y sin `--config`. Las dependencias
Python del entorno siguen siendo necesarias; el bundle incluye nuestro código.

Primero valida un solo dispositivo. `launch_xla_audited.py` es un piloto opcional
para repartir **modelos independientes** entre dispositivos; no usa all-reduce
ni cambia batch global. Consume memoria host por worker; no presupongas speedup 8×.
El lanzamiento multiproceso aún requiere validación en la TPU real; empieza con
un panel pequeño y abandona esta optimización si consume el presupuesto de control.
Para dos GPU, procesos independientes con `CUDA_VISIBLE_DEVICES` y
`--workers 2 --worker-id 0/1` comparten el output únicamente si receta/backend
coinciden. No ejecutes dos workers con el mismo índice sobre los mismos jobs.

## 8. Presupuesto y recuperación del historial

La confirmación fiel necesita más modelos que un panel simple. Reserva:

| Actividad | GPU | TPU |
|---|---:|---:|
| Piloto y reproducibilidad | 2 h | 2 h |
| Cribado features | 0 h | 4 h |
| Pesos limpios vs control | 6 h | 0 h |
| Confirmación temporal / seeds adicionales | 8 h | 12 h |
| Final e inferencia | 10 h | 0 h |
| Reserva | 4 h | 2 h |
| Total máximo | 30 h | 20 h |

Son techos, no instrucciones de agotarlos. Con ~17 min/modelo, los 40 modelos
de confirmación estiman 11,3 h TPU: divide en sesiones/reanuda antes del timeout.
No confundas tiempo transcurrido de varias sesiones con suma de cuota GPU.
Mide la tarifa/cuota real de la configuración de aceleradores que use Kaggle.
Si el piloto es más lento, recorta experimentos opcionales; no elimines el control
ni el bloque externo para meter más candidatos. Feature builds/rescore son CPU.

Para reparar métricas antiguas sin reentrenar, descarga el tensor por modelo y
su JSON de folds de los outputs históricos:

```bash
python src/kaggle/rescore_legacy_oof.py \
  --models /ruta/oof_models_kfold455s5.npy --y /ruta/full_y.npy \
  --months /ruta/full_month.npy --fold-report /ruta/metrics_kfold455s5.json \
  --out runs/rescore_v15
```

En paneles multibrazo añade `--arm base` o `--arm x30`. El script no promedia
variantes entre sí y muestra cuántos modelos predicen realmente cada fila.
Esto repara el promedio de validación; no crea una evaluación externa retroactiva.

Entrega a Diego: configs congelados, manifests/hashes, cuota consumida, modelos
completos, ambos scores, deltas por bloque/mes/seed, resultado procesado, fallos
de reproducibilidad y recomendación de candidato. No prometas un rango en LB.
