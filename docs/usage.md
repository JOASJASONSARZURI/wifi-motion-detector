# Guía de uso

## Requisitos

Python 3.10 o posterior. El paquete necesita NumPy; las pruebas usan pytest.
La interfaz visual usa Tkinter, incluido normalmente con Python para Windows.

Abra PowerShell en la carpeta del repositorio que contiene `pyproject.toml`
(`wifi-motion-detector`, no la carpeta contenedora):

```powershell
cd C:\ruta\al\detector_wifi\wifi-motion-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Si la instalación indica que no encuentra `pyproject.toml`, compruebe que está
en la carpeta del proyecto. Para confirmar los comandos disponibles:

```powershell
wifi-motion --help
wifi-motion run --help
wifi-motion eval --help
```

## Abrir la interfaz visual

```powershell
python -m wifi_motion.gui
```

La aplicación abre en modo demostración. **MODO DEMO - SYNTHETIC** y los
resultados de `DemoSource` son generados por código, no son mediciones de Wi-Fi
ni una conexión a hardware. En la pantalla, INICIAR y DETENER controlan el
sondeo; EXPORTAR permite guardar el log visible. CALIBRAR está deshabilitado en
esta versión.

## Generar datos sintéticos

Para recorrer calibración, entrenamiento, ejecución y evaluación sin hardware:

```powershell
wifi-motion generate-synthetic --out data\synthetic --sessions 4 --seed 7
New-Item -ItemType Directory -Force artifacts | Out-Null
```

Las sesiones generadas se identifican como `syn-000` en adelante. El comando
también guarda `syn-calibration-absent.jsonl`, una grabación etiquetada solo
como `absent`.

## Calibrar una línea base

```powershell
wifi-motion calibrate --input data\synthetic\syn-calibration-absent.jsonl --node-id node-1 --out artifacts\baseline.json
```

La calibración requiere una grabación únicamente `absent` y suficientes
ventanas válidas. Las opciones de ventana disponibles son `--window-s`,
`--hop-s`, `--min-samples` y `--max-gap-s`. Consulta [calibration.md](./calibration.md)
para preparar grabaciones reales y revisar la línea base.

## Entrenar un modelo softmax

```powershell
wifi-motion train --train data\synthetic\syn-000.jsonl,data\synthetic\syn-001.jsonl --val data\synthetic\syn-002.jsonl --out artifacts\model.json
```

`train` usa train para ajustar el modelo y val para validación. No uses las
sesiones reservadas para test durante el entrenamiento.

## Ejecutar una grabación con la CLI

Clasificador por umbral:

```powershell
wifi-motion run --input data\synthetic\syn-003.jsonl --classifier threshold --baseline artifacts\baseline.json
```

Clasificador softmax:

```powershell
wifi-motion run --input data\synthetic\syn-003.jsonl --classifier softmax --model artifacts\model.json
```

Ambos comandos escriben resultados JSONL a stdout y estadísticas a stderr.
`unknown` significa que no hubo datos válidos para decidir; no significa
`absent`. Las opciones de ventana y suavizado disponibles se muestran con
`wifi-motion run --help`.

## Evaluar clasificadores

Reserva sesiones distintas para train, val y test. Este ejemplo evalúa una sola
vez `syn-003` como test:

```powershell
wifi-motion eval --train data\synthetic\syn-000.jsonl,data\synthetic\syn-001.jsonl --val data\synthetic\syn-002.jsonl --test data\synthetic\syn-003.jsonl --baseline artifacts\baseline.json --model artifacts\model.json --out-dir artifacts\evaluation
```

El comando compara el baseline mayoritario, el clasificador por umbral y el
modelo softmax, por ventana y en flujo. Con `--val`, ajusta los umbrales usando
solo validación. Escribe `report.json`, `report.md` y agrega una ejecución a
`eval_log.jsonl`. Si ya aparecen los mismos IDs de test con las mismas rutas de
modelo, advierte que se reutilizó el test; la advertencia no detiene la
ejecución. No interpretes datos `SYNTHETIC` como rendimiento medido.
Consulta [evaluation.md](./evaluation.md) para el protocolo y las definiciones.

## Reproducir un JSONL desde la GUI

`ReplaySource` está disponible para reproducir una grabación con un clasificador
ya configurado. No hay selector de archivo ni conexión automática desde la
ventana actual; para usarlo, pasa la fuente al crear `App`. Ejemplo con el
clasificador por umbral:

```powershell
python -c "import tkinter as tk; from wifi_motion.gui.app import App; from wifi_motion.gui.replay_source import ReplaySource; root=tk.Tk(); source=ReplaySource(r'data\synthetic\syn-003.jsonl', classifier='threshold', baseline_path=r'artifacts\baseline.json'); App(root, source); root.mainloop()"
```

Para un modelo softmax, sustituye la creación de la fuente por:

```python
ReplaySource(r"data\synthetic\syn-003.jsonl", classifier="softmax", model_path=r"artifacts\model.json")
```

`ReplaySource` procesa la grabación en un hilo y entrega resultados a la GUI.
La fuente conserva `evidence = "SYNTHETIC"` o `"MEASURED"` según los metadatos;
en esta versión, el banner visible de la app se activa automáticamente para
`DemoSource`, no para grabaciones reproducidas. Una sesión marcada `synthetic`
no debe presentarse como una medición real.

## Pruebas

Desde la raíz del proyecto:

```powershell
python -m pytest -q
```

Las pruebas Tkinter se omiten cuando el entorno no dispone de pantalla.
`wifi-motion record` todavía no captura hardware: el subcomando informa que no
está implementado.
