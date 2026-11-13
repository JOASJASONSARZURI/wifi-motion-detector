# Ejecución con Docker

Docker permite ejecutar la interfaz de línea de comandos sin instalar Python ni
las dependencias del proyecto en el equipo. La imagen instala el paquete desde
`pyproject.toml` y usa Python 3.12. Los comandos se ejecutan como procesos
puntuales; los resultados se conservan en las carpetas del proyecto montadas en
el contenedor.

## Requisitos

- Docker Desktop en Windows o Docker Engine con el complemento Docker Compose.
- Ejecutar los comandos desde la raíz del repositorio, donde están
  `Dockerfile` y `compose.yaml`.

## Construir la imagen

```powershell
docker compose build
```

Compose instala la dependencia de ejecución NumPy y crea la imagen local
`wifi-motion-detector:local`.

## Ejecutar comandos

La imagen inicia `wifi-motion` directamente. Los argumentos que siguen al
nombre del servicio se pasan a la CLI:

```powershell
docker compose run --rm wifi-motion --help
docker compose run --rm wifi-motion generate-synthetic --out /app/data/synthetic --sessions 4 --seed 7
```

`--rm` elimina el contenedor al terminar, no los archivos de salida. El volumen
`./data` del equipo se monta en `/app/data` y `./artifacts` en `/app/artifacts`.
Por eso, las grabaciones generadas se encuentran en `data/synthetic` en el
repositorio y se pueden reutilizar en ejecuciones posteriores.

Ejemplo de calibración con la grabación sintética creada arriba:

```powershell
docker compose run --rm wifi-motion calibrate --input /app/data/synthetic/syn-calibration-absent.jsonl --node-id node-1 --out /app/artifacts/baseline.json
```

La GUI (`python -m wifi_motion.gui`) no se incluye en esta imagen: requiere
Tkinter y acceso a un servidor gráfico, que no forman parte del uso estándar de
Docker Desktop. Para usar la GUI, sigue la guía local en [usage.md](./usage.md).
`wifi-motion record` tampoco captura hardware en esta versión.

## Reconstruir y limpiar

Tras cambiar el código o sus dependencias, reconstruye la imagen:

```powershell
docker compose build
```

Para borrar la imagen local:

```powershell
docker image rm wifi-motion-detector:local
```

Esto no borra `data/` ni `artifacts/`, porque son carpetas del equipo, no datos
del contenedor.
