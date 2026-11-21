# NavIL
Navegación autónoma de robots móviles mediante aprendizaje por imitación en ROS 2.

## Paquete ROS 2 `il_data`

El módulo está en `src/il_data` y usa el sistema de paquetes `ament_python`.
El esquema común está en `src/il_data/il_data/schema.py`. Cada muestra contiene
90 lecturas LiDAR remuestreadas, la distancia y el ángulo del objetivo en el
marco del robot, y las velocidades lineal/angular que sirven como etiqueta
experta. Los datasets se guardan en JSONL: un objeto JSON validable por línea.

### Probar sin ROS 2

Con Python 3.10 o posterior, desde la raíz del repositorio:

```powershell
python -m unittest discover -s src\il_data\test -t src\il_data -v
$env:PYTHONPATH = "src\il_data"
python -m il_data.generate_fake_dataset --output dataset_ficticio.jsonl --samples 100 --seed 42
```

El generador es determinista para una misma semilla y permite probar el
esquema, la escritura y la lectura sin tener Gazebo ni Nav2 instalados. Estos
datos son sintéticos: sirven para comprobar el pipeline, no para entrenar ni
evaluar el modelo.

### Compilar e instalar el paquete ROS 2

Desde la raíz del repositorio, en un entorno con ROS 2 Jazzy, `colcon` y las
dependencias del paquete:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select il_data
source install/setup.bash
```

El paquete declara sus dependencias ROS en `src/il_data/package.xml` y registra
los comandos mediante `setup.py`; no es necesario copiar los scripts a otro
paquete. Antes de entregar, sustituye el correo de ejemplo
`navil-team@example.com` en `package.xml` y `setup.py` por un contacto real del
equipo.

### Grabar un rosbag2

```bash
ros2 run il_data navil-record --output bags/episodio_001
```

Por defecto graba `/scan` (`sensor_msgs/msg/LaserScan`), `/cmd_vel`
(`geometry_msgs/msg/Twist`), `/amcl_pose`
(`geometry_msgs/msg/PoseWithCovarianceStamped`) y `/goal_pose`
(`geometry_msgs/msg/PoseStamped`). Los nombres se pueden cambiar con
`--scan-topic`, `--cmd-vel-topic`, `--pose-topic` y `--goal-topic`.
La pose localizada y la meta deben expresarse en el mismo marco de referencia.
Detén la grabación con Ctrl+C; no se sobrescriben carpetas de salida existentes.

### Convertir el rosbag a dataset

```bash
ros2 run il_data navil-bag-to-dataset bags/episodio_001 data/episodio_001.jsonl
```

El conversor empareja cada mensaje de `/cmd_vel` con las observaciones previas
más recientes. LiDAR y pose localizada deben tener como máximo 0.5 s de
antigüedad (`--max-age` permite cambiarlo); si faltan datos o están desfasados,
el comando se descarta y se informa. La conversión falla explícitamente si
faltan tópicos, sus tipos son distintos, las poses usan marcos incompatibles o
no se puede formar ninguna muestra.

### Comprobar un dataset

`il_data.dataset_io.cargar_jsonl(ruta)` carga y valida cada registro con
`MuestraNavegacion`. Los errores incluyen el número de línea para localizar los
datos incorrectos.

Para ejecutar las pruebas desde el workspace ROS 2:

```bash
colcon test --packages-select il_data
colcon test-result --verbose
```
