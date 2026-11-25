# NavIL

Navegación autónoma de robots móviles mediante aprendizaje por imitación en ROS 2.

## Paquete ROS 2 `il_data`

El paquete está en `ros2_ws/src/il_data` y usa `ament_python`. Su salida de
entrenamiento sigue el contrato de la integrante 3: un archivo comprimido
`.npz` con exactamente estas claves:

| Clave | Forma | Tipo | Contenido |
|---|---:|---|---|
| `scan` | `(N, 90)` | `float32` | Distancias LiDAR en metros |
| `goal` | `(N, 2)` | `float32` | Distancia en metros y ángulo en radianes |
| `v` | `(N,)` | `float32` | Velocidad lineal experta en m/s |
| `w` | `(N,)` | `float32` | Velocidad angular experta en rad/s |
| `episode_id` | `(N,)` | entero | Identificador numérico del recorrido |

### Convención LiDAR

Los 90 rayos de `scan` están separados 4 grados y ordenados desde el frente
en sentido antihorario: `scan[0]` apunta delante, `scan[22]` a 88 grados
(izquierda), `scan[45]` detrás y `scan[67]` a 268 grados (derecha). El
conversor utiliza `angle_min` e `angle_increment` del mensaje ROS para
reordenar/interpolar los rayos; no presupone que el primer elemento del mensaje
sea el frontal. Por defecto, el frame de `/scan` debe estar alineado con el
frente del robot; si tiene un giro, indícalo con `--scan-yaw-offset` en
radianes. El resultado se guarda en metros, limitado por defecto a 12 m.
Las lecturas `NaN` o `+inf` se sustituyen por el máximo; `-inf`, por cero.
Los sectores que no cubra un LiDAR de campo de visión parcial también se
rellenan con el máximo.

El objetivo se calcula en el marco del robot y usa ángulo positivo hacia la
izquierda (REP-103). Las velocidades expertas conservan las unidades físicas
del mensaje `/cmd_vel`.

## Probar sin ROS 2

Con Python 3.10 o posterior y NumPy, desde la raíz del repositorio:

```powershell
python -m unittest discover -s ros2_ws\src\il_data\test -t ros2_ws\src\il_data -v
$env:PYTHONPATH = "ros2_ws\src\il_data"
python -m il_data.generate_fake_dataset --output dataset_ficticio.npz --samples 100 --seed 42 --episode-id 0
```

El generador es determinista para una misma semilla. Los datos son sintéticos:
sirven para probar el formato, no para entrenar ni evaluar el modelo. El
archivo se puede inspeccionar así:

```powershell
python -c "import numpy as np; d=np.load('dataset_ficticio.npz'); print({k: (d[k].shape, d[k].dtype) for k in d.files})"
```

## Compilar e instalar con ROS 2 Jazzy

Desde la raíz del repositorio:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select il_data
source install/setup.bash
```

`ros2_ws/src/il_data/package.xml` declara las dependencias ROS y NumPy. `setup.py`
registra los comandos `navil-record`, `navil-bag-to-dataset` y
`navil-generate-fake-dataset`.

## Grabar y convertir datos reales

Graba los tópicos `/scan`, `/cmd_vel`, `/amcl_pose` y `/goal_pose`:

```bash
ros2 run il_data navil-record --output bags/episodio_001
```

Detén la grabación con Ctrl+C. Los nombres de tópicos se pueden cambiar con
`--scan-topic`, `--cmd-vel-topic`, `--pose-topic` y `--goal-topic`.

Convierte el bag a NPZ. El identificador debe ser un entero no negativo y único
para cada recorrido que luego se combine para entrenar:

```bash
ros2 run il_data navil-bag-to-dataset bags/episodio_001 data/episodio_001.npz --episode-id 1
```

LiDAR y pose deben tener como máximo 0,5 s de antigüedad (`--max-age` permite
cambiarlo). El rango para lecturas sin retorno es 12 m por defecto y puede
cambiarse con `--scan-max-range` (en ese caso, configura el mismo
`scan_max_range` en el contrato de entrenamiento). El conversor falla explícitamente si faltan
tópicos, sus tipos son incorrectos, las poses y la meta usan marcos distintos,
o no se puede formar ninguna muestra.

Para comprobar un archivo desde Python, usa
`il_data.dataset_io.cargar_npz(ruta)`: valida las cinco claves, sus formas y
tipos y devuelve los arrays para el entrenamiento.

## Pruebas en ROS 2

```bash
colcon test --packages-select il_data
colcon test-result --verbose
```
