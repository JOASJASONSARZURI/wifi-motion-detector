# Guía para probar con un ESP32 físico

Esta guía describe el flujo futuro desde una captura CSI hasta el detector.
El proyecto todavía no lee el puerto serial ni captura hardware: `wifi-motion
record` responde «aún no implementado». `run` y la GUI procesan archivos.

## 1. Qué hace falta

- Un ESP32 compatible con el ejemplo CSI de Espressif, un cable USB de datos y
  un computador con el entorno ESP-IDF requerido por la versión elegida de
  `esp-csi`.
- Para recibir CSI basta empezar con un ESP32 ejecutando `csi_recv` y un emisor
  de tráfico Wi-Fi que el receptor pueda observar. Puede generarse tráfico
  desde el computador hacia el router, por ejemplo con ping, si el montaje y el
  firmware lo permiten.
- Un segundo ESP32 ejecutando `csi_send` es opcional; puede dar una fuente de
  tráfico más controlada. No es requisito para probar la primera captura.
- Un terminal serial que permita guardar la salida a un archivo. No se requiere
  `pyserial` para este flujo por archivos.

## 2. Preparar y probar el firmware

1. Identifica el modelo exacto del ESP32 y consulta el repositorio y las
   instrucciones de Espressif para `esp-csi`:
   [github.com/espressif/esp-csi](https://github.com/espressif/esp-csi).
   Escoge el ejemplo `csi_recv` y una versión de ESP-IDF compatible con ese
   modelo y con esa versión del ejemplo.
2. Sigue las instrucciones del ejemplo para configurar, compilar y cargar el
   firmware. El puerto, la velocidad serial, las opciones de configuración y
   los comandos dependen del modelo y de la versión; no uses valores
   supuestos de esta guía.
3. Abre el monitor serial usando la configuración indicada por el firmware.
   Genera tráfico Wi-Fi y confirma que aparecen líneas CSI. Si el ejemplo no
   imprime `CSI_DATA` o el campo CSI no está presente, el conversor de este
   proyecto no podrá procesar esa salida sin adaptaciones.
4. Guarda la salida original sin reformatearla. Conserva una copia para
   verificar el formato y registrar qué firmware y configuración se usaron.

Un segundo ESP32 con `csi_send` puede usarse como transmisor alternativo. Para
la primera prueba, no cambies simultáneamente firmware, canal, ubicación y
fuente de tráfico: anota cada configuración para que las sesiones se puedan
reproducir.

## 3. Antes de convertir el log

El archivo `scripts/esp32_to_jsonl.py` es un esbozo, no una herramienta validada
contra el formato de tu firmware. En el código actual, `csv.reader` entrega una
fila y se accede a `row[3]` para RSSI (cuarto campo contando desde 1) y `row[18]`
para `local_timestamp` (decimonoveno campo contando desde 1). Además, asume que:

- el campo CSI es una secuencia de enteros alternados `[imag, real, ...]`;
- cada pareja corresponde a una subportadora y todas las líneas válidas tienen
  el mismo número de subportadoras.

Estas posiciones y convenciones pueden variar entre versiones. **No uses el
JSONL resultante para calibrar ni reportar métricas hasta contrastar esas
suposiciones con varias líneas completas de tu propio log y corregir/probar el
conversor para ese firmware.** No compartas el log públicamente si contiene
direcciones MAC u otros identificadores; conserva la estructura de columnas al
redactar una copia.

El esbozo descarta líneas sin informar cuántas descartó; solo imprime cuántas
escribió. También fija un timestamp de época artificial que comienza en
`1_700_000_000.0` segundos y suma el tiempo relativo del primer
`local_timestamp`; no es la hora real de la captura. Antes de usarlo, hay que
comprobar que el contador local no se reinicia ni desborda durante la sesión,
que produce tiempos estrictamente crecientes y que las amplitudes y el RSSI se
leen de las columnas correctas. Un reinicio, wrap, duplicado o dato mal
parseado debe causar un error o descarte visible, no una muestra aparentemente
válida.

Después de validar el formato y el conversor, la forma de invocarlo es:

```powershell
python scripts\esp32_to_jsonl.py --log data\raw\captura.txt --out data\recordings\absent-01.jsonl --session-id sala-absent-01 --node-id node-1 --label absent --rate-hz TASA_MEDIDA --environment "sala y condiciones descritas"
```

Reemplaza `TASA_MEDIDA` por la tasa observada y usa una sesión e identificador
nuevos para cada grabación. El conversor asigna una sola etiqueta a todo el
archivo; detén o separa la captura al cambiar de estado. Los valores escritos
por una captura física se marcan `synthetic=false`, pero solo son evidencia
**MEASURED** cuando el método, el firmware, el montaje y la reproducción quedan
documentados.

## 4. Crear las grabaciones propias

1. Mantén fijos la ubicación y orientación de los nodos, el canal, el enlace y
   la configuración de captura durante una tanda. Anota el montaje y las
   condiciones en los metadatos; no incluyas nombres ni datos personales.
2. Graba una sesión `absent` con la zona vacía y el montaje definitivo para
   calibrar. La guía de calibración recomienda aproximadamente dos minutos en
   sala vacía; comprueba que la tasa real proporcione suficientes ventanas
   válidas.
3. Graba sesiones etiquetadas `present_still` y `present_moving` con criterios
   consistentes. Haz sesiones separadas y usa `session_id` distintos para
   entrenamiento, validación y prueba. No asignes a una misma grabación más de
   una etiqueta con el conversor actual.
4. Conserva el log original, el JSONL convertido, la etiqueta, el montaje, la
   tasa observada y la configuración del firmware juntos. Protege las
   grabaciones: sus marcas de tiempo y etiquetas pueden revelar presencia.

## 5. Calibrar, entrenar y evaluar

Instala el proyecto desde la carpeta que contiene `pyproject.toml`:

```powershell
python -m pip install -e ".[dev]"
```

Calibra la línea base solo con la grabación `absent`:

```powershell
wifi-motion calibrate --input data\recordings\absent-01.jsonl --node-id node-1 --out artifacts\node-1-baseline.json
```

Revisa el número de ventanas usadas y `floored_features`. Si la tasa medida
causa ventanas inválidas, ajusta las opciones `--window-s`, `--hop-s`,
`--min-samples` o `--max-gap-s` según los datos; no tomes valores sintéticos
como configuración validada para el dispositivo.

Entrena softmax usando únicamente sesiones de entrenamiento y validación:

```powershell
wifi-motion train --train data\recordings\train-01.jsonl,data\recordings\train-02.jsonl --val data\recordings\val-01.jsonl --out artifacts\model.json
```

Evalúa una sola vez en sesiones reservadas para test, disjuntas de train y val:

```powershell
wifi-motion eval --train data\recordings\train-01.jsonl,data\recordings\train-02.jsonl --val data\recordings\val-01.jsonl --test data\recordings\test-01.jsonl --baseline artifacts\node-1-baseline.json --model artifacts\model.json --out-dir artifacts\evaluation
```

Inspecciona `report.json` y `report.md`; el comando también registra las
ejecuciones en `eval_log.jsonl` y advierte si se repiten IDs de test con las
mismas rutas de modelo. No presentes los resultados como rendimiento
reproducido en hardware si no conservas y documentas el procedimiento.

Ejecuta una grabación guardada con uno de los clasificadores:

```powershell
wifi-motion run --input data\recordings\test-01.jsonl --classifier threshold --baseline artifacts\node-1-baseline.json
wifi-motion run --input data\recordings\test-01.jsonl --classifier softmax --model artifacts\model.json
```

También puedes abrir la GUI con `python -m wifi_motion.gui` y elegir
**REPRODUCIR JSONL**. Esa opción reproduce el archivo seleccionado; no abre el
puerto serial ni funciona en vivo. `unknown` significa que no hay datos válidos
para decidir, no que la zona esté ausente.

## 6. Expectativas y límites

Con los valores predeterminados indicados por el proyecto
(`--window-s 4`, `--min-samples 50`, `--max-gap-s 0.15`) se necesitan alrededor
de 13 Hz o más; la tasa real debe medirse y podría requerir ajustar el
ventaneo. Un flujo irregular puede producir muchos `unknown`. La línea base y
el modelo sintéticos no son utilizables como calibración de hardware real.

Cambios en ubicación, mobiliario, hardware, canal o entorno pueden exigir
recalibración y nuevas sesiones. `present_still` es la clase más débil; no
interpretes ausencia de movimiento como prueba suficiente de presencia quieta.
El sistema usa un enlace y no estima pose, identidad ni ubicación. La
adaptación automática de la línea base no está activa.

Para operar en vivo haría falta implementar y probar una fuente serial que
lea el ESP32, valide el formato y alimente al detector de forma incremental.
Eso no lo proporciona el flujo de archivos descrito aquí.
