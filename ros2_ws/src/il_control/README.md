# il_control: Módulo 4 (integración ROS 2 y evaluación)

Rama: `feature/il-control` · Estado: **ciclo 2** (inferencia con `il_policy`, preprocesado idéntico al dataset, runner de episodios en bucle cerrado con métricas)

El paquete ejecuta la política aprendida como nodo ROS 2, decide si al robot le llega Nav2 (experto) o la red (política), nunca las dos a la vez, aplica una capa de seguridad antes de `/cmd_vel` y mide episodios de navegación en bucle cerrado.

```
 episode_runner ──► /navil/goal_pose ──┐            /scan ──────────────┬───────────────┐
   │  (o navil_expert)                 ▼                                ▼               ▼
   │                          il_policy_node ──► /cmd_vel_policy ──┐  ┌─ seguridad ─┐
   │                           (20 Hz, TF)                         ├─►cmd_vel_mux ───────► /cmd_vel ─► TB4
   │  Nav2 (remap) ───────────────────────────► /cmd_vel_expert ───┘  mode: expert|policy|stop
   └─► mode (servicio de parámetros) ──────────────────────────────────┘
   └─► /navil/episode_event + episodes.csv + summary.json
```

## Estructura

| Fichero | Contenido |
|---|---|
| `il_control/observation.py` | **Preprocesado** LiDAR → 90 valores y objetivo → `[dist, ángulo]`. Modo `interp`, réplica exacta de `il_data`, y modo `min`. Funciones puras |
| `il_control/safety.py` | `SafetyFilter`: límites, watchdogs de LiDAR y de comando, parada/frenado por proximidad y rampa |
| `il_control/policies.py` | Backends con `predict(obs) → (v, w)`: `il_policy` (adaptador de `ILPolicy`), `mock_reactive`, `mock_goal` y `zero` |
| `il_control/episodes.py` | Escenarios (formato de `navil_expert`), decisión de fin de episodio, métricas y CSV/JSON. Sin ROS |
| `il_control/policy_node.py` | Nodo de inferencia → `/cmd_vel_policy` |
| `il_control/cmd_vel_mux.py` | Selector exclusivo + seguridad. **Único publicador de `/cmd_vel`** |
| `il_control/episode_runner.py` | Ejecuta rutas en bucle cerrado con la política o con Nav2 y guarda las métricas |
| `il_control/fake_sim.py`, `sim_world.py` | Simulador cinemático (stub de Gazebo): `/scan`, `/odom`, TF y colisiones |
| `config/il_control.yaml` | Parámetros de `il_policy_node` y `cmd_vel_mux` |
| `config/scenarios.yaml` | Rutas: las del almacén (las mismas que en `navil_expert`) y las del stub |
| `config/mock_sim.yaml` | Mundo del stub |
| `launch/il_control.launch.py` | Nodo de inferencia + mux |
| `launch/closed_loop.launch.py` | Inferencia + mux + runner (Gazebo). Se cierra solo al terminar |
| `launch/mock_test.launch.py` | Lo mismo sobre el stub, sin Gazebo ni Nav2 |
| `test/` | 57 tests con pytest |

## Instalación (dentro del contenedor)

```bash
cd ~/ros2_ws
colcon build --packages-select il_control --symlink-install
source install/setup.bash
```

Dependencias: paquetes estándar de ROS 2 Jazzy y numpy. Con `policy_type:=il_policy` hacen falta además el paquete `il_policy` compilado en el mismo workspace y **PyTorch**. Para comprobarlo:

```bash
python3 -c "import torch; print(torch.__version__)"
# Si falla: pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
```

## Uso sin Gazebo (stub)

```bash
ros2 launch il_control mock_test.launch.py                       # conduce el "experto" mock
ros2 param set /cmd_vel_mux mode policy                          # conmuta en caliente
ros2 launch il_control mock_test.launch.py mode:=policy policy_type:=mock_goal   # la seguridad la frena

# Ciclo 2: episodios automáticos con métricas
ros2 launch il_control mock_test.launch.py runner:=true scenario:=mock_circuito
ros2 launch il_control mock_test.launch.py runner:=true scenario:=mock_cruce controller:=expert
```

## Uso con Gazebo (ciclo 2)

**1. Simulación y localización.** Hay que seguir `INSTRUCCIONES_ENTORNO.md` de Lucas, terminales 1 y 2: Gazebo y `localization.launch.py` con `warehouse.yaml`. **Nav2 solo hace falta para `controller:=expert`.** AMCL necesita la pose inicial: se puede dar con *2D Pose Estimate* en RViz o pasándola al runner con `-p initial_pose:="x,y,yaw"`. Si el robot arranca en la base de carga, conviene desacoplarlo primero:

```bash
ros2 action send_goal /undock irobot_create_msgs/action/Undock {}
```

**2. Checkpoint del modelo.** Mientras no haya un modelo entrenado con datos reales, sirve uno de prueba de Elena, generado con su script desde su rama:

```bash
USE_SOURCE=1 ./scripts/train_policy_dummy.sh          # -> models/mlp_dummy/best.pt
```

**3. Episodios con la política:**

```bash
ros2 launch il_control closed_loop.launch.py scenario:=ruta_validada repetitions:=5 \
     policy_type:=il_policy model_path:=$HOME/ros2_ws/models/mlp_dummy/best.pt
```

**4. Las mismas rutas con Nav2** (necesita Nav2 lanzado y su salida remapeada a `/cmd_vel_expert`):

```bash
ros2 launch il_control closed_loop.launch.py scenario:=ruta_validada repetitions:=5 controller:=expert
```

Resultados en `~/navil_results/<controller>_<escenario>_<fecha>/`:
- `episodes.csv`: una fila por episodio. Se escribe al terminar cada uno, así no se pierde nada si se cae Gazebo.
- `summary.json`: métricas agregadas.

Durante la ejecución:

```bash
ros2 topic echo /navil/episode_event        # mismo JSON que navil_expert (+ outcome y metrics al final)
ros2 topic echo /il_control/status          # modo, motivo de seguridad, holgura
ros2 topic hz /cmd_vel                      # ~20 Hz, la salida del mux
```

### Qué mide cada episodio

| Columna | Definición |
|---|---|
| `outcome` | `success` (distancia al objetivo < `goal_tolerance`, o éxito de Nav2 en modo experto), `timeout`, `stuck`, `collision` o `aborted` (Nav2) |
| `time_s` | Tiempo simulado desde que se publica el objetivo |
| `path_length_m` | Longitud recorrida, integrada con la TF `odom → base_link` |
| `path_efficiency` | Avance neto hacia el objetivo / longitud recorrida (1 = línea recta) |
| `collisions` | Contactos del bumper (`/hazard_detection`, tipo BUMP) o `/il_control/sim_collision` (stub). Un choque = un evento por cada 2 s |
| `estop_events` | Veces que la capa de seguridad entró en parada por proximidad (casi colisiones) |
| `min_obstacle_dist_m` | Distancia mínima del LiDAR a un obstáculo durante el episodio (medida desde el centro del sensor) |
| `inference_ms_mean`, `_p95`, `inference_hz` | Latencia de `predict()` en `il_policy_node`. En modo experto es la de la política ejecutándose en sombra |

`timeout = timeout_base + timeout_per_meter · distancia_recta` (por defecto 30 s + 10 s/m). `stuck`: avanza menos de 0.10 m en 15 s.

## Decisiones de diseño (para la memoria y la defensa)

* **Nodo selector propio en vez de `twist_mux`.** `twist_mux` elige por prioridad y por quién publicó último. Aquí se necesita exclusión explícita fijada por el experimento, y que la fuente inactiva se ignore aunque siga publicando.
* **La política corre siempre; el mux elige.** La red puede ejecutarse en *modo sombra* mientras conduce Nav2, comparando `/cmd_vel_policy` con `/cmd_vel_expert` en el mismo instante. Eso permite medir el MAE online (ciclo 3) y observar el *covariate shift* sin arriesgar el robot.
* **Control a 20 Hz con timer fijo.** Es la frecuencia del `controller_server` de Nav2, que es la que define el dataset: `il_data` genera una muestra por cada `cmd_vel` del experto. Así la red decide al mismo ritmo al que se tomaron las decisiones que imita. El periodo no depende del LiDAR (50 Hz en la simulación) y el mux sigue publicando ceros aunque las entradas se detengan.
* **El preprocesado de inferencia es el del dataset (`scan_reduction: interp`).** Se ha portado `remuestrear_lidar` de `il_data` y un test comprueba que da exactamente el mismo resultado. Si entrenamiento e inferencia preprocesan distinto (*train/serve skew*), el modelo recibe entradas que nunca vio. Tiene una limitación conocida: al muestrear puntualmente cada 4°, un obstáculo de menos de 4° puede no aparecer (hay un test que lo documenta). La capa de seguridad, que trabaja con todos los rayos, cubre ese caso. El modo `min` existe por si el equipo decide regenerar el dataset así.
* **Runner propio en vez de `trajectory_generator`.** El de Lucas manda los objetivos a Nav2, que con la política al mando seguiría intentando controlar y lanzaría recuperaciones. El runner publica el objetivo y los eventos en el mismo formato que `navil_expert`. En modo experto usa la acción de Nav2, de modo que **las dos opciones se evalúan con las mismas rutas, límites, seguridad y métricas**.
* **Longitud con odometría y objetivo con `map`.** Las correcciones de AMCL producen saltos en `map` que inflarían la longitud recorrida; la odometría es suave a corto plazo. Para la distancia al objetivo sí se usa `map`, porque el objetivo está en ese marco.
* **QoS:** `SensorDataQoS` para `/scan` (un scan perdido se sustituye por el siguiente) y fiable con profundidad 10 para los comandos. `/il_control/mode` es *transient local*.
* **Watchdogs:** sin LiDAR reciente (0.5 s) no se puede garantizar la parada por obstáculo, así que el robot se para. Si la fuente activa deja de publicar (0.5 s), también.
* **Proximidad por pasillo**, un rectángulo del ancho del robot en la dirección de avance. Solo anula la componente lineal y deja girar, con frenado progresivo entre `slow_distance` y `stop_distance`. Es **la misma capa para Nav2 y para la red**, para que la comparación sea justa.
* **Objetivo y yaw del LiDAR por TF.** Así no depende de `/amcl_pose`, que solo se publica cuando AMCL actualiza, ni de saber a mano cómo va montado el `rplidar_link`.

## Contrato con los otros módulos

| Con | Qué | Estado |
|---|---|---|
| `il_policy` (Elena) | `ILPolicy.load(path).predict({'scan', 'goal'}) → (v, w)`. El nº de rayos y el rango máximo se leen del checkpoint | ✅ compatible (test contra `contract.py`) |
| `il_data` (María) | Mismo preprocesado del LiDAR y del objetivo al grabar y al inferir | ✅ réplica exacta de `remuestrear_lidar` (test de equivalencia) |
| `navil_expert` (Lucas) | Objetivo en `/navil/goal_pose`, eventos en `/navil/episode_event`, escenarios en el mismo YAML | ✅ compatible |
| `navil_expert` (Lucas) | Salida final de Nav2 remapeada a `/cmd_vel_expert`; tipo `Twist` o `TwistStamped` | ⚠️ pendiente (solo para `controller:=expert` y para el modo sombra) |
