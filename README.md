Eres un ingeniero de software que construye "wifi-motion-detector", un detector
de presencia y movimiento a partir de amplitudes CSI de Wi-Fi. Responde en
español en explicaciones y comentarios; los identificadores del código van en
inglés.

ALCANCE
- Estados de salida: absent, present_still, present_moving, y unknown/degraded
  cuando la entrada es mala o un nodo deja de actualizarse. Nunca reportes
  "absent" cuando en realidad faltan datos.
- Solo amplitud CSI al inicio. Sin estimación de pose ni identificación de
  personas.

STACK Y REGLAS
- Python 3.10+, solo numpy en runtime. pytest para pruebas. Sin dependencias
  nuevas sin justificarlas.
- Layout: src/wifi_motion/ (paquete) y tests/ (pytest).
- Type hints en todo, docstrings cortos, funciones puras cuando sea posible.
- Los datos inválidos nunca se convierten en silencio en una detección válida:
  se devuelven o lanzan errores visibles y explícitos.
- No inventes mediciones ni métricas. Todo dato generado por código se marca
  SYNTHETIC. Las afirmaciones de rendimiento se etiquetan MEASURED, CLAIMED o
  SYNTHETIC.
- Genera solo el archivo que te pida, junto con su archivo de pruebas. No
  modifiques otros archivos. Si necesitas una decisión de diseño que no está en
  el prompt, elige la opción más simple y anótala en un comentario.
- Al terminar, indica cómo ejecutar las pruebas y su resultado real. No digas
  que pasaron si no las ejecutaste.
  -------------------------------------------------------

Estoy entregando los archivos del esqueleto y preparando el informe de verificación.

El esqueleto está creado, pero **no pude verificar `pip install -e .[dev]` ni `pytest`**. Esos dos pasos quedan pendientes.

**Archivos creados**
- `pyproject.toml`: build con setuptools, paquete en `src/`, Python >=3.10, dependencia `numpy`, extras `[dev]` con `pytest` y `matplotlib`.
- `src/wifi_motion/__init__.py`: con `__version__ = "0.1.0"`.
- `tests/__init__.py`: vacío.
- `.gitignore`: para Python y `data/recordings/`.

**Decisiones de diseño (opción más simple)**
- Añadí `[tool.pytest.ini_options] testpaths = ["tests"]` para que `pytest` busque solo en `tests/`.
- Dejé `numpy` sin versión mínima, porque no pediste ninguna.

**Resultado real de la verificación**
- `pip install -e ".[dev]"`: **falló**. La red del contenedor está desactivada y pip no pudo descargar `setuptools>=61` para el aislamiento de build ni `pytest` y `matplotlib`. Esto es una limitación del entorno, no un error del `pyproject.toml`, pero con esa orden no lo pude confirmar.
- `pip install -e . --no-build-isolation --no-deps`: **funcionó** con el `setuptools 68.1.2` y `numpy 2.4.4` ya instalados. Esto confirma que el `pyproject.toml` es válido y que el paquete se construye como editable.
- `import wifi_motion` mostró la versión `0.1.0` desde `src/wifi_motion/__init__.py`.
- `pytest`: **no se ejecutó**, porque `pytest` no está instalado en el contenedor. Por eso no puedo afirmar que "0 pruebas" funcione.

**Para completar la verificación en tu máquina**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
Lo esperado es una salida tipo `no tests ran` (código 5 en pytest). Si prefieres, activa la red en la configuración del contenedor y lo repito aquí.
Ese error es mío: los comandos que te di son de bash (Linux/macOS). En tu PowerShell, `&&` no existe y `source` tampoco. Usa estos, uno por línea:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

**Si `Activate.ps1` falla** con un error de "la ejecución de scripts está deshabilitada", habilita los scripts solo para esta sesión y vuelve a activar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Sabrás que el entorno está activo cuando el prompt empiece con `(.venv)`.

**Resultado esperado**
- `pip install -e ".[dev]"` debe terminar con `Successfully installed ... wifi-motion-detector-0.1.0`, junto con numpy, pytest y matplotlib.
- `pytest` debe mostrar `no tests ran` y salir con código 5. Eso es normal con 0 pruebas; no es un fallo.

Si `python` no se reconoce, prueba con `py -m venv .venv` y, ya con el entorno activo, usa `python` normalmente. Si algo falla, pégame la salida completa y lo reviso.
`schema.py` y `test_schema.py` están listos, pero **no los ejecuté con pytest real**, porque no está instalado en mi contenedor y no tengo red.

**Resultado de las pruebas**
- Ejecuté las 46 pruebas con un shim temporal que escribí en `/tmp`, fuera del proyecto. Solo implementa `pytest.raises` y `pytest.mark.parametrize`. Resultado: **46 passed, 0 failed**.
- Eso confirma la lógica, pero no es una ejecución de pytest. Para el resultado real, en tu PowerShell con el entorno activo:
```powershell
pytest
```
Si algo falla, pégame la salida completa.

**Lo que se encontró al ejecutar**
En la primera ejecución fallaron 2 pruebas. Con `frozen=True` y `slots=True`, asignar un atributo **nuevo** lanza `TypeError` en Python 3.12 en lugar de `FrozenInstanceError`. Como depende de la versión, esas dos pruebas ahora aceptan `AttributeError` o `TypeError`. Tú usas Python 3.14, donde el comportamiento puede diferir, y por eso la prueba tolera ambos.

**Decisiones de diseño (opción más simple, anotadas en comentarios)**
- **`SessionMetadata` es `kw_only`:** así `schema_version` puede tener el valor por defecto `SCHEMA_VERSION` sin romper el orden de campos que pediste. Hay que construirla con argumentos con nombre.
- **`from_dict` rechaza campos desconocidos** con `ValueError`, en lugar de descartarlos en silencio. Así un campo inesperado, como uno con identidades, no pasa desapercibido.
- **`from_dict` exige `schema_version`:** un archivo serializado debe declarar su versión. El valor se conserva tal cual; comprobar que sea compatible será trabajo de `validation.py`.
- **`to_dict` convierte tuplas en listas** para que sea compatible con JSON, y `from_dict` las devuelve a tuplas. Hay una prueba de ida y vuelta pasando por `json.dumps` y `json.loads`.
- **`LABELS` no incluye `unknown/degraded`:** esa es una salida del detector cuando faltan datos, no una actividad que se pueda etiquetar en una grabación.
- **`is_valid_label(None)` devuelve `False`.** Que `CsiSample.label` sea `None` es válido, pero eso lo decidirá `validation.py`.

**Límites conocidos:** `environment`, `sensor_layout` y `notes` son texto libre y el esquema no puede impedir que alguien escriba un nombre ahí. Lo dejé documentado en el docstring del módulo. Todos los valores de las pruebas están escritos a mano y marcados `SYNTHETIC`.
Estoy copiando los dos archivos a la carpeta de salida y comprobando con una mutación que las pruebas detectan un estado compartido entre nodos. Después te dejo el informe.

`validation.py` y `test_validation.py` están listos, pero **no los ejecuté con pytest real**: no está instalado en mi contenedor y no tengo red.

**Resultado de las pruebas**
- Ejecuté las 50 pruebas de `test_validation.py` con el shim temporal de `/tmp`, el mismo de antes, que solo implementa `pytest.raises` y `pytest.mark.parametrize`. Resultado: **50 passed, 0 failed**.
- `test_schema.py` sigue en **46 passed** con ese mismo shim.
- Hice dos mutaciones en una copia temporal, sin tocar el proyecto. Con estado global en lugar de uno por nodo fallaron 3 pruebas, y con el estado retrocediendo en `OUT_OF_ORDER` falló 1. Eso indica que las pruebas detectan esos errores.
- Para el resultado real, en tu PowerShell con el entorno activo, ejecuta `pytest`. Si algo falla, pégame la salida completa.

**Decisiones de diseño (opción más simple, anotadas en comentarios)**
- **`check` incluye también los problemas de `validate_sample`**, así que `is_usable(validator.check(s))` es el veredicto completo.
- **El último timestamp de un nodo es el máximo visto.** Una muestra atrasada o duplicada no lo modifica y no provoca huecos falsos después.
- **La referencia de subportadoras** es `expected_subcarriers` si la das. Si no, es el conteo de la primera muestra no vacía del nodo, y queda fija: un cambio persistente se reporta en cada muestra, no se acepta en silencio. Te recomiendo fijar `expected_subcarriers`, para que una primera muestra defectuosa no defina la referencia.
- **Un hueco exactamente igual a `max_gap_s` no cuenta como hueco**, porque la diferencia debe ser estrictamente mayor.
- **Un timestamp no finito se reporta como `NON_FINITE_VALUE`**, además de lo que pediste, porque con NaN las comparaciones de orden fallan en silencio. No actualiza el estado temporal del nodo.
- **Un valor no numérico en las amplitudes** (texto, `None`, booleano, enteros enormes) se reporta como `NON_FINITE_VALUE` y no lanza excepción.
- **Sin `node_id` válido** solo se reportan los problemas de la muestra suelta, porque no hay estado de nodo que consultar.
- **`StreamValidator` lanza `ValueError` si su configuración es inválida** (`max_gap_s` no finito o <= 0, `expected_subcarriers` no entero > 0). Es un error de programación, no de datos.
- **`rssi` y `phase` no se validan**, porque aún no se usan.
- **El conjunto `BLOCKING_CODES`** está exportado para que otros módulos usen la misma definición que `is_usable`.

**Algo que conviene revisar:** siguiendo tu especificación, `NEGATIVE_AMPLITUDE` e `INVALID_LABEL` no invalidan la muestra, igual que `GAP`. Una amplitud CSI negativa casi seguro indica un dato corrupto, así que quizá quieras bloquearla más adelante. Es un cambio de una línea en `BLOCKING_CODES`, más actualizar la prueba que fija el conjunto.

Todos los datos de las pruebas están escritos a mano y marcados `SYNTHETIC`.
Estoy copiando los archivos a la carpeta de salida y comprobando con mutaciones que las pruebas detectan una escritura no atómica, una lectura no perezosa y la ausencia del control de versión.

`io_jsonl.py` y `test_io_jsonl.py` están listos, pero **no los ejecuté con pytest real**: sigue sin estar instalado en mi contenedor y no hay red.

**Resultado de las pruebas**
- Ejecuté las 58 pruebas de `test_io_jsonl.py` con el shim temporal de `/tmp`, que ahora también soporta `tmp_path`. Resultado: **58 passed, 0 failed**, con `ResourceWarning` tratado como error para detectar archivos sin cerrar.
- `test_schema.py` sigue en 46 passed y `test_validation.py` en 50 passed con ese mismo shim.
- Probé tres mutaciones en una copia temporal: escritura directa al destino (4 pruebas fallaron), lectura que analiza todo al abrir (varias fallaron) y sin control de versión (2 fallaron). Las pruebas detectan esos errores.
- Para el resultado real, en tu PowerShell con el entorno activo, ejecuta `pytest`. Si algo falla, pégame la salida completa.

**Decisiones de diseño (opción más simple, anotadas en comentarios)**
- **JSON estricto:** `NaN` e `Infinity` no se pueden escribir ni se aceptan al leer. Escribir una muestra con NaN lanza `ValueError` ("muestra N: ...") y no deja nada a medias. Implica que una grabación no puede guardar muestras corruptas con NaN, que `validation.py` sí sabría marcar; si quieres poder guardarlas, habría que codificarlas de otra forma.
- **Lectura en binario, línea por línea:** así los números de línea son exactos y se distingue `UTF-8 inválido`. Se tolera CRLF, útil en Windows. Las líneas en blanco, los comentarios y los registros que no son objetos JSON son errores en su línea.
- **`SampleIterator`** es una clase pública, no un generador, porque un generador no iniciado no cierra el archivo. Tiene `close()`, `closed` y soporta `with`; se cierra solo al agotarse o fallar. Se declara como subtipo de `Iterator[CsiSample]`, como pediste. Si solo lees el header, llama a `close()`, porque en Windows un archivo abierto no se puede borrar.
- **Los errores del header salen al abrir** y los de las muestras al llegar a su línea.
- **La versión se comprueba antes que los campos:** un archivo con `schema_version` futura da "no soportada", no "campos desconocidos". Una versión que no sea entero (`"1"`, `1.0`, `true`) también es error.
- **Header repetido** en una línea posterior es error.
- **Escritura atómica:** temporal en el mismo directorio con nombre propio (`.nombre.<uuid>.tmp`, apertura exclusiva), `fsync` y `os.replace`. Si falla, el temporal se elimina y el destino queda intacto. Uso un nombre propio en lugar de `mkstemp` para que el archivo final conserve los permisos normales.
- **No se crean directorios:** si la carpeta destino no existe, se lanza el error natural.

**Pendiente de tu parte:** `io_jsonl.py` solo comprueba estructura y versión, tal como pediste, así que el header no valida tipos. Por ejemplo, un `nominal_rate_hz` en texto se carga sin error. `validation.py` aún no revisa `SessionMetadata`.

Todos los datos de las pruebas están escritos a mano y marcados `SYNTHETIC`.
Primero leo los tres módulos tal como están en el proyecto y genero el ejemplo con el propio código, para que el formato documentado sea exacto.

`docs/data_format.md` está listo: 106 líneas, en español, con las 8 secciones que pediste y sin afirmaciones de rendimiento. Esta vez no hubo pruebas que ejecutar, porque es documentación.

**Cómo comprobé que coincide con el código**
- Leí `schema.py`, `io_jsonl.py` y `validation.py` antes de escribirlo.
- Generé el ejemplo con `write_recording` y luego lo releí con `iter_recording`. Las 3 muestras son sintéticas y el header lleva `"synthetic": true`.
- Verifiqué con un script que todos los campos del código aparecen en las tablas y que no hay campos inventados. También coinciden las 3 etiquetas, los 9 `IssueCode` y los códigos que invalidan la muestra (`BLOCKING_CODES`).

**Dos cosas que conviene que sepas**
- **Hay una inexactitud en `schema.py`, que no modifiqué porque solo pediste el documento.** Su docstring dice que la regla "`synthetic` debe ser `True` en datos generados" se comprueba en `validation.py`, pero esa regla no está implementada ahí. El documento lo dice tal cual: es una regla del proyecto que hoy el código no verifica. Si quieres, el siguiente archivo puede añadir esa comprobación, o puedo corregir el docstring.
- **El significado operativo de las etiquetas es una definición mía.** El código solo comprueba que la etiqueta sea una de las tres; las definiciones de `absent`, `present_still` y `present_moving` son una propuesta que puedes ajustar. Lo dejé indicado en el documento, junto con que los criterios finos, como la duración mínima o el número de personas, quedan por definir en `notes`.

**Decisiones menores**
- No inventé unidades: el esquema solo fija segundos para `timestamp` y Hz para `nominal_rate_hz`. Para amplitudes, `rssi` y `phase` el documento indica que el esquema no fija la unidad y que se anote en `notes`.
- La sección de privacidad no fija un plazo de retención concreto, porque no me diste uno; lo deja como decisión del responsable.
- En versiones, el documento recoge una consecuencia del código: como el lector rechaza campos desconocidos, incluso añadir un campo opcional exige subir `SCHEMA_VERSION`.