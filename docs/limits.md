# Límites y estado de la evidencia

## 1. Estado de la evidencia

| Afirmación | Etiqueta | Motivo |
|---|---|---|
| El pipeline y las pruebas funcionan con datos generados por código. | SYNTHETIC | Los tests y las sesiones generadas son sintéticos; demuestran ejecución del software, no exactitud física. |
| Rendimiento medido por esta implementación en hardware real. | SYNTHETIC | El repositorio no contiene capturas reales registradas ni una reproducción documentada; no hay métricas MEASURED. |
| Métricas del archivo externo `references/wifi_densepose_results.csv`. | CLAIMED | La guía lo describe como especificaciones y métricas reportadas, sin muestras CSI ni evidencia independiente para reproducirlas; no se usa como dataset ni como resultado de esta implementación. |

`MEASURED` solo corresponde a una afirmación reproducida con hardware real y
con el método de reproducción documentado. La etiqueta de una prueba sintética
no se convierte en evidencia medida por pasar el test.

## 2. Qué hace y qué no hace

El clasificador devuelve `absent`, `present_still` o `present_moving`. El
detector puede devolver `unknown` cuando una ventana no permite una detección
válida; no hay una clase `degraded` separada en el código. El sistema no estima
pose, no identifica personas, no cuenta personas y no localiza posiciones.

## 3. Límites físicos y de datos

- CSI depende del hardware, frecuencia, ubicación de transmisores y receptores,
  geometría, mobiliario, paredes, interferencia y tasa de muestreo; un cambio
  puede requerir recalibración.
- RSSI por sí solo no equivale a CSI ni necesariamente aporta información
  suficiente para discriminar movimiento de forma robusta.
- Un solo enlace puede no separar varias personas ni inferir posición.
- No se promete detección fiable a través de paredes.
- Las ventanas usan muestras reales, sin interpolar ni rellenar. Huecos,
  ventanas con pocas muestras y características imposibles de extraer limitan
  lo que puede clasificarse.

## 4. Límites del modelo

- `present_still` del clasificador por umbral es evidencia débil: la ausencia de
  movimiento no implica presencia quieta.
- Las puntuaciones y `confidence` no son probabilidades calibradas.
- Softmax y los umbrales dependen de datos y ajustes limitados; no se garantiza
  generalización entre sesiones, ambientes, hardware o ubicaciones.
- Las características y bandas de frecuencia son definiciones del proyecto,
  con valores iniciales que no provienen de mediciones universales.

## 5. Límites de esta implementación

- `wifi-motion record` no está implementado: el comando indica que requiere
  hardware y el formato real de la fuente CSI. No hay lectura directa de hardware.
- La adaptación automática de la línea base no está activa. `detector.py`
  conserva el TODO: “adaptar baseline solo durante periodos quietos confirmados”.
- El proyecto define 15 características propias en `FEATURE_NAMES`; no son una
  medida directa de pose ni una descripción universal del canal.
- `FeatureConfig` usa bandas iniciales de movimiento y respiración; el código
  indica que deben ajustarse con datos medidos y no son umbrales universales.

## 6. Datos malos y `unknown`

Si una ventana tiene huecos, muestras insuficientes o falla la extracción de
características/clasificación, el detector emite `unknown` con un motivo, no
`absent`. `check_stale` también puede emitir `unknown` con motivo `stale_node`
cuando se invoca para un nodo obsoleto. Una muestra individual rechazada por
validación se descarta y contabiliza; no produce una detección `absent` ni
necesariamente genera un resultado `unknown`.

## 7. Privacidad y uso responsable

No almacene identidades ni datos personales en IDs, notas o metadatos. Obtenga
consentimiento antes de capturar espacios ocupados, limite la retención y
proteja las grabaciones: etiquetas y marcas temporales pueden revelar presencia
y actividad. No use este prototipo para vigilancia de personas.

## 8. Etiquetar afirmaciones nuevas

Use **MEASURED** solo con reproducción documentada en hardware real, **CLAIMED**
para resultados externos no reproducidos y **SYNTHETIC** para datos generados
por código; nunca presente métricas sintéticas o reclamadas como medidas.