# Calibración de la línea base

## 1. Qué es la línea base

Una `Baseline` describe las características de un solo nodo en su ubicación
instalada mientras la sala está vacía. Se guarda una mediana y un MAD
(desviación absoluta mediana, sin escalar) por característica. Un nodo requiere
su propia línea base.

La mediana y el MAD son estadísticas robustas: un movimiento puntual en menos
de la mitad de las ventanas no arrastra la estimación como podría hacerlo la
media y la desviación estándar. No sustituyen una captura realmente vacía.

## 2. Grabación de calibración

- Deje los nodos en la ubicación y orientación definitivas.
- Asegúrese de que nadie esté en la zona monitoreada ni cerca de ella durante
  toda la captura.
- Mantenga puertas y ventanas en su estado habitual de operación.
- Grabe el tiempo suficiente para obtener al menos `min_windows` ventanas
  válidas de características. `fit_baseline` requiere por defecto 30; ventanas
  inválidas o con error de características no se cuentan.
- `calibrate` copia el campo `environment` del header a `environment_note`.
  Describa allí las condiciones relevantes: disposición, mobiliario, puertas
  y hora de captura. No incluya nombres ni datos personales.

El comando `calibrate` solo acepta una grabación etiquetada enteramente como
`absent` y selecciona las muestras del `--node-id` indicado.

## 3. Ejecución

Con la CLI instalada `wifi-motion`, por ejemplo:

```console
wifi-motion calibrate --input data/sala-vacia.jsonl --node-id node-1 --out artifacts/node-1-baseline.json
```

La CLI usa las opciones de ventana disponibles en `calibrate`: `--window-s`,
`--hop-s`, `--min-samples` y `--max-gap-s`. Solo ventanas de calidad `OK` con
características extraídas se usan para ajustar la línea base. El resultado
indica el número de ventanas usadas y cuántas se descartaron.

## 4. Qué revisar después

- Confirme que se usó el `node_id` esperado y que `n_windows` refleja suficientes
  ventanas válidas.
- Revise `Baseline.floored_features`: enumera las características cuyo MAD
  original fue cero y se sustituyó por `MAD_EPSILON`. Un MAD cero significa que
  la característica no varió en la calibración; queda con una escala artificial
  muy pequeña, por lo que cualquier desvío puede producir una puntuación enorme.
  No confíe en esas características sin revisarlas.

## 5. Archivo de línea base

`save_baseline` escribe JSON estricto con estos campos:

| Campo | Contenido |
|---|---|
| `baseline_version` | Versión del formato (`BASELINE_VERSION`) |
| `node_id` | Nodo al que corresponde la línea base |
| `feature_names` | Nombres y orden de las características |
| `median` | Mediana por característica |
| `mad` | MAD sin escalar por característica |
| `n_windows` | Ventanas usadas al calibrar |
| `created_at` | Instante de creación en segundos de época Unix |
| `environment_note` | Nota de entorno copiada del header `environment` |

La versión actual es `BASELINE_VERSION = 1`. Un RSSI ausente puede representarse
como `null` en `median` y `mad` de `mean_rssi`; no se inventa una medida.

## 6. Actualización en línea

`update_baseline` existe, pero solo debe aplicarse cuando el periodo sea quieto.
La función recibe explícitamente `quiet`; si es falso devuelve la misma línea
base. Si es verdadero, desplaza la mediana mediante EMA; no cambia el MAD,
`n_windows` ni `created_at`. La quietud puede comprobarse con `is_quiet`, usando
un `z_max` y `features_to_check` elegidos con datos medidos.

El detector actual **no** actualiza la línea base automáticamente. En
`detector.py` permanece este TODO: “adaptar baseline solo durante periodos
quietos confirmados.” Si `quiet` se calcula mal, `update_baseline` puede hacer
derivar la línea base hacia la presencia; aunque `alpha` sea pequeño, la deriva
es real y puede normalizar gradualmente una presencia como si fuera el estado
vacío.

## 7. Cuándo recalibrar

Vuelva a calibrar si cambia la ubicación de los nodos, el mobiliario, el
hardware, el canal Wi-Fi o el entorno. Cada línea base solo describe las
condiciones de su calibración.

## 8. Límites

La calibración describe solo esa sala y ese enlace. `z_max` y las características
usadas para decidir quietud dependen del hardware y del entorno: deben ajustarse
con datos medidos y no son umbrales universales. La desviación respecto a la
línea base no es una probabilidad ni demuestra por sí sola que la sala esté vacía.