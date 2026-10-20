# Formato de grabación de wifi-motion-detector

> Derivado de `src/wifi_motion/schema.py`, `io_jsonl.py` y `validation.py`. Si el código cambia, este documento debe actualizarse.

## 1. Propósito y versión del esquema

Define cómo se guardan las grabaciones de amplitudes CSI de Wi-Fi (una sesión por archivo) para reproducirlas, etiquetarlas y validarlas. Solo se guarda amplitud CSI; la fase es opcional y todavía no se usa.

Versión actual: `SCHEMA_VERSION = 1`, escrita en el header de cada archivo.

## 2. Estructura del archivo (JSONL)

Una línea por registro, UTF-8, cada línea termina en `\n`: la línea 1 es el `header` y las siguientes son registros `sample`.

- No hay comentarios, líneas en blanco ni otros tipos de registro: toda línea no reconocida es un `RecordingFormatError` con su número de línea y motivo.
- JSON estricto: `NaN` e `Infinity` no se escriben ni se aceptan.
- Al escribir se incluyen todos los campos (los opcionales vacíos van como `null`); al leer, los opcionales pueden faltar. Un campo desconocido es error.
- La escritura es atómica (archivo temporal y renombrado) y la lectura es perezosa. La lectura tolera finales de línea CRLF.

Ejemplo (SYNTHETIC, valores escritos a mano, `"synthetic": true`):

```jsonl
{"type":"header","schema_version":1,"session_id":"sesion-sintetica-001","environment":"sala-101","sensor_layout":"nodo n1 a 1 m de altura, 3 m frente al nodo n2","nominal_rate_hz":10.0,"notes":"SYNTHETIC: valores escritos a mano","synthetic":true}
{"type":"sample","timestamp":1700000000.0,"node_id":"n1","amplitudes":[1.0,1.5,2.0,1.5],"rssi":null,"phase":null,"label":"absent"}
{"type":"sample","timestamp":1700000000.1,"node_id":"n1","amplitudes":[1.1,1.4,2.1,1.6],"rssi":-50.0,"phase":null,"label":"absent"}
{"type":"sample","timestamp":1700000000.2,"node_id":"n2","amplitudes":[3.0,2.5,2.0,2.5],"rssi":null,"phase":[0.0,0.1,0.2,0.3],"label":"present_still"}
```

## 3. Campos

**Header** (`SessionMetadata`, metadatos no sensibles del montaje):

| Nombre | Tipo | Obligatorio | Descripción | Unidad |
|---|---|---|---|---|
| `type` | texto | sí | Siempre `"header"` | — |
| `schema_version` | entero | sí (al leer) | Versión del esquema; debe ser `1` | — |
| `session_id` | texto | sí | Identificador de la sesión | — |
| `environment` | texto | sí | Texto libre, p. ej. `"sala-101"` | — |
| `sensor_layout` | texto | sí | Texto libre: distancias, alturas | — |
| `nominal_rate_hz` | número | sí | Tasa de muestreo nominal | Hz |
| `notes` | texto | no (`""`) | Notas libres | — |
| `synthetic` | booleano | no (`false`) | Debe ser `true` en todo dato generado por código | — |

**Muestra** (`CsiSample`):

| Nombre | Tipo | Obligatorio | Descripción | Unidad |
|---|---|---|---|---|
| `type` | texto | sí | Siempre `"sample"` | — |
| `timestamp` | número | sí | Instante de la muestra, época Unix con fracciones | s |
| `node_id` | texto | sí | Identificador del sensor/nodo | — |
| `amplitudes` | lista de números | sí | Una amplitud por subportadora | (*) |
| `rssi` | número o `null` | no | Intensidad de señal recibida | (*) |
| `phase` | lista de números o `null` | no | Fase por subportadora; no se usa todavía | (*) |
| `label` | texto o `null` | no | Etiqueta de actividad, solo en grabaciones (sección 4) | — |

(*) El esquema no fija esta unidad; anótela en `notes`. El lector comprueba estructura y versión, no los tipos de cada valor ni el contenido de la señal (eso es de `validation.py`). La regla de `synthetic` es del proyecto: hoy el código no la verifica.

## 4. Etiquetas válidas

`LABELS = ("absent", "present_still", "present_moving")`; `is_valid_label` acepta exactamente esas tres.

| Etiqueta | Significado operativo |
|---|---|
| `absent` | Ninguna persona en el área monitoreada. |
| `present_still` | Una o más personas presentes sin desplazamientos apreciables (p. ej. sentadas o de pie, quietas). |
| `present_moving` | Una o más personas presentes y en movimiento (p. ej. caminando). |

Son definiciones operativas para etiquetar; el código solo comprueba que la etiqueta sea una de las tres. Los criterios finos (duración mínima, número de personas) no están definidos: documéntelos en `notes`. `unknown/degraded` no es una etiqueta: es una salida del detector cuando faltan datos o son malos, y no se usa en grabaciones.

## 5. Errores y problemas de datos

**Errores de formato** (`RecordingFormatError`, con línea y motivo): archivo vacío o sin header, `schema_version` no soportada o no entera, JSON inválido o UTF-8 inválido, `type` ausente o desconocido, header repetido, campos faltantes o desconocidos, `NaN`/`Infinity`.

**Problemas de datos** (`Issue` de `validation.py`): se detectan al pasar las muestras por `validate_sample` (muestra suelta) o `StreamValidator.check` (flujo, estado por `node_id`; incluye los de muestra suelta). Un archivo con estos problemas se lee sin error.

| `IssueCode` | Significado | ¿Invalida la muestra? |
|---|---|---|
| `EMPTY_AMPLITUDES` | `amplitudes` vacía | sí |
| `NON_FINITE_VALUE` | `timestamp` o alguna amplitud es NaN, infinita o no numérica | sí |
| `NEGATIVE_AMPLITUDE` | Alguna amplitud es menor que 0 | no |
| `SUBCARRIER_COUNT_MISMATCH` | Distinto número de subportadoras que `expected_subcarriers` o que la primera muestra del nodo | sí |
| `DUPLICATE_TIMESTAMP` | Mismo `timestamp` que el último del nodo | sí |
| `OUT_OF_ORDER` | `timestamp` anterior al último visto del nodo | sí |
| `GAP` | Salto mayor que `max_gap_s` respecto a la muestra anterior del nodo | no, pero debe quedar visible |
| `MISSING_NODE_ID` | `node_id` vacío o que no es texto | sí |
| `INVALID_LABEL` | `label` presente y distinta de las tres válidas | no |

`is_usable(issues)` es falso si hay algún código que invalida. Regla del proyecto: los datos inválidos se reportan; nunca se convierten en silencio en una detección válida.

## 6. Privacidad

- **Sin identidades:** ningún campo debe contener identidades de personas. `node_id` identifica un sensor; `session_id`, `environment`, `sensor_layout` y `notes` son texto libre y no deben incluir nombres ni datos personales (el esquema no puede comprobarlo).
- **Consentimiento:** obtenerlo de quienes estén en el espacio antes de capturar en espacios ocupados.
- **Retención limitada:** fijar de antemano un plazo de conservación y borrar las grabaciones al vencer. Este documento no fija el plazo.
- **Sin datos innecesarios:** guardar solo lo que el proyecto usa; no añadir notas con detalles personales.
- Etiquetas y marcas de tiempo revelan presencia y actividad en un lugar y momento: trate las grabaciones reales como sensibles. No se versionan: `data/recordings/` está en `.gitignore`.

## 7. Política de versiones

- El lector exige `schema_version == SCHEMA_VERSION`. Cualquier otra, mayor o menor, da `RecordingFormatError` en la línea 1 ("no soportada"), antes de revisar el resto de campos.
- Como el lector rechaza campos desconocidos, todo cambio de estructura requiere subir `SCHEMA_VERSION`, incluso añadir un campo opcional; también renombrar o quitar campos y cambiar tipo, unidad o significado.
- No hay migraciones implementadas. Si la versión cambia, hay que escribir una conversión explícita y probada y actualizar este documento; los archivos antiguos no se reinterpretan en silencio.

## 8. Aviso sobre datos SYNTHETIC

Los datos marcados SYNTHETIC (ejemplos y pruebas, escritos a mano o generados por código) solo sirven para probar el formato y el código: no demuestran exactitud de campo. Este documento no hace afirmaciones de rendimiento; cualquiera futura debe etiquetarse MEASURED, CLAIMED o SYNTHETIC.