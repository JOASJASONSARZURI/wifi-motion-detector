# Evaluación de clasificadores y del flujo

Las grabaciones se etiquetan con `absent`, `present_still` o `present_moving`.
`unknown` es una salida del detector, no una etiqueta real.

## 1. Qué se mide

`eval` compara cada clasificador por ventana y en flujo.

- **Matriz de confusión:** filas de clase real; columnas `absent`, `present_still`, `present_moving`, `unknown`.
- **Precision:** aciertos de una clase / predicciones de esa clase; sin predicciones queda `null`.
- **Recall:** aciertos de una clase / ventanas reales de esa clase.
- **F1:** media armónica de precision y recall. `unknown` nunca acierta: es error de la clase real.
- **Macro F1:** media de F1 solo entre clases con soporte real; las clases sin soporte tienen métricas `null`.
- **False alarm rate por ventana:** ventanas reales `absent` predichas `present_*` / todas las ventanas reales `absent`; sin `absent`, no está definido.
- **False negative rate por ventana:** ventanas reales `present_*` predichas `absent` / todas las ventanas reales `present_*`; sin presencia, no está definido.
- **Unknown rate por ventana:** predicciones `unknown` / total de ventanas evaluadas.
- **Falsas alarmas de flujo:** resultados `present_*` mientras la última muestra etiquetada del nodo es `absent`; su tasa divide por todos los resultados cuya etiqueta real es `absent`.
- **Unknown rate de flujo:** resultados `unknown` / todos los resultados alineados con una muestra etiquetada.
- **Tamaño/procedencia:** sesiones, IDs y ventanas válidas de test por clase.

Ventanas inválidas o no etiquetables se descartan al construir el dataset; el
reporte cuenta las ventanas que sí se evalúan.

## 2. Protocolo

- Separe `train`, `val` y `test` por `session_id`; `eval` rechaza IDs solapados.
- Para probar generalización, separe también por ambiente o disposición de sensores; sesiones del mismo ambiente pueden no representar un ambiente nuevo.
- Ajuste el preprocesamiento con train y parámetros con validación, nunca test. `eval` ajusta umbrales solo con `val`; sin val usa `--motion-z` y `--presence-z`.
- El softmax ya debe estar entrenado: `eval` lo carga sin reajustarlo. Use test una sola vez, después de fijar modelo y parámetros.
- `eval_log.jsonl` agrega fecha, IDs de test y rutas absolutas del baseline/modelo por ejecución. Al hallar los mismos IDs y modelos, avisa en `stderr` que el test ya se usó y el resultado deja de ser una evaluación única; no bloquea la ejecución.

## 3. Baselines comparados

- **Mayoritaria (`baseline`):** predice la clase más frecuente en ventanas de train; desempata por el orden de `CLASS_STATES`.
- **Umbral (`threshold`):** aplica la línea base cargada; si hay `val`, `tune_thresholds` elige `motion_z` y `presence_z` por macro F1 en validación.
- **Softmax (`softmax`):** carga el modelo entrenado indicado por `--model`.

Los tres se evalúan también en el detector completo con la configuración de
ventana y suavizado indicada en la CLI.

## 4. Latencia y transiciones

Para cada cambio real por sesión y nodo, latencia es el tiempo desde el primer
sample con la nueva etiqueta hasta el primer resultado cuyo `state` sea esa
etiqueta. Debe ocurrir antes del siguiente cambio real del nodo.

Si no aparece ese estado antes del siguiente cambio (o del final de la sesión),
el cambio es `missed` y no aporta latencia. Se resumen latencias observadas con
media, mediana y máximo por tipo de transición.

`--transition-tolerance-s` excluye de las métricas por ventana resultados a una
distancia estrictamente menor a la tolerancia de cualquier cambio real e informa
el recuento excluido. No cambia latencias ni tasas de falsas alarmas/`unknown`
del flujo.

## 5. Ejecución con datos sintéticos

Comandos reales de la CLI instalada `wifi-motion`. Cada sesión generada tiene un
ID independiente; se asignan sesiones distintas a los splits.

```console
wifi-motion generate-synthetic --out data/synthetic --sessions 4 --seed 7
wifi-motion calibrate --input data/synthetic/syn-calibration-absent.jsonl --node-id node-1 --out artifacts/baseline.json
wifi-motion train --train data/synthetic/syn-000.jsonl,data/synthetic/syn-001.jsonl --val data/synthetic/syn-002.jsonl --out artifacts/model.json
wifi-motion eval --train data/synthetic/syn-000.jsonl,data/synthetic/syn-001.jsonl --val data/synthetic/syn-002.jsonl --test data/synthetic/syn-003.jsonl --baseline artifacts/baseline.json --model artifacts/model.json --out-dir artifacts/evaluation
```

`eval` escribe `report.json`, `report.md` y agrega `eval_log.jsonl` en `--out-dir`.
Lo sintético prueba el código, no valida el sistema con hardware real.

## 6. Etiquetas de evidencia

- **SYNTHETIC:** todas las sesiones de test tienen `synthetic=true`; la separabilidad es por construcción y no valida hardware real.
- **MEASURED:** ninguna sesión de test tiene `synthetic=true`; no certifica por sí sola el diseño experimental ni la calidad de etiquetas.
- **CLAIMED:** afirmación de rendimiento citada/comunicada, no producida por `eval`; debe incluir su fuente y no sustituye evidencia medible y reproducible.

No mezcle sesiones sintéticas y medidas: se evita atribuir a mediciones reales
resultados que reflejan la construcción de datos sintéticos.

## 7. Limitaciones

- Un conjunto pequeño o poco diverso no representa cambios de ocupación, actividad, ambiente ni captura.
- La señal depende de hardware, ubicación, orientación y disposición; un resultado no se transfiere automáticamente a otro montaje.
- Un solo enlace no describe otros enlaces ni garantiza cobertura espacial.
- RSSI no es CSI: RSSI no reemplaza amplitudes CSI por subportadora.
- Un cambio de entorno o ubicación puede requerir recalibrar la línea base y reajustar parámetros con datos permitidos.
- Separar sesiones reduce contaminación, pero no garantiza generalización entre sesiones, ambientes o hardware.
- `present_still` es la clase más débil: ausencia de movimiento no demuestra por sí sola presencia quieta.
- Datos sintéticos prueban reproducibilidad e integración, no exactitud en condiciones reales.

## 8. Plantilla de resultados reales

Rellene una fila por clasificador y ejecución; no use resultados sintéticos como
resultados de campo.

| Fecha | Evidencia | Sesiones | Ventanas | Clasificador | F1 macro | Falsas alarmas | Falsos negativos | Latencia mediana |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |
