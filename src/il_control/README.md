# il_control: Módulo 4 (integración ROS 2 y evaluación)

Rama: `feature/il-control` · Estado: **ciclo 1** (modelo mock, simulador stub, selector y seguridad operativos)

El paquete ejecuta la política aprendida como nodo ROS 2, decide si al robot le llega Nav2 (experto) o la red (política), nunca las dos a la vez, y aplica una capa de seguridad antes de `/cmd_vel`.

```
                     /scan ─────────────┬──────────────────────────┐
                                        ▼                          ▼
 /goal_pose ──► il_policy_node ──► /cmd_vel_policy ──┐      ┌─ seguridad ─┐
   + TF          (10 Hz, timer)                      ├─► cmd_vel_mux ──────► /cmd_vel ──► robot
 Nav2 (remap) ─────────────────► /cmd_vel_expert ────┘  mode: expert|policy|stop  (20 Hz)
```

## Estructura

| Fichero | Contenido |
|---|---|
| `il_control/observation.py` | **Preprocesado** LiDAR → 90 sectores y objetivo → `[dist, ángulo]`. Funciones puras, misma convención que `il_policy/contract.py` |
| `il_control/safety.py` | `SafetyFilter`: límites, watchdogs de LiDAR y de comando, parada/frenado por proximidad y rampa de aceleración |
| `il_control/policies.py` | Backends con `predict(obs) → (v, w)`: `mock_reactive`, `mock_goal`, `zero` e `il_policy` (adaptador de `ILPolicy`) |
| `il_control/policy_node.py` | Nodo de inferencia → `/cmd_vel_policy` |
| `il_control/cmd_vel_mux.py` | Selector exclusivo + seguridad. **Único publicador de `/cmd_vel`** |
| `il_control/fake_sim.py`, `sim_world.py` | Simulador cinemático (stub de Gazebo): `/scan`, `/odom`, TF y `/goal_pose` |
| `config/il_control.yaml` | Parámetros de los dos nodos |
| `config/mock_sim.yaml` | Escenario de prueba sin Gazebo |
| `launch/il_control.launch.py` | Lanza el módulo 4 (para usar con Gazebo + Nav2) |
| `launch/mock_test.launch.py` | Prueba completa del ciclo 1 sin Gazebo ni Nav2 |
| `test/` | 41 tests con pytest (preprocesado, seguridad, políticas, simulador y bucle cerrado) |

## Instalación (dentro del contenedor)

```bash
cd ~/ros2_ws            # raíz del repo
colcon build --packages-select il_control --symlink-install
source install/setup.bash
```

Solo depende de paquetes estándar de ROS 2 Jazzy y de numpy. PyTorch y `il_policy` hacen falta únicamente con `policy_type:=il_policy`.

## Uso: ciclo 1 (sin Gazebo)

```bash
ros2 launch il_control mock_test.launch.py                 # conduce el "experto" mock
ros2 param set /cmd_vel_mux mode policy                    # conmuta a la política en caliente
ros2 param set /cmd_vel_mux mode stop                      # parada

# Política "ciega" (no esquiva): la capa de seguridad tiene que frenarla
ros2 launch il_control mock_test.launch.py mode:=policy policy_type:=mock_goal

ros2 topic echo /il_control/status                         # modo, motivo de seguridad, holgura
ros2 topic echo /il_policy_node/inference_time_ms
```

Para probar los watchdogs, matad `fake_expert` o `fake_sim` con el robot en marcha: el mux pasa a `cmd_timeout` o `scan_timeout` y publica ceros.

Tests: `python3 -m pytest src/il_control/test` (no hace falta compilar) o `colcon test --packages-select il_control`.

## Uso con Gazebo + Nav2 (ciclo 2)

```bash
ros2 launch il_control il_control.launch.py                                   # Nav2 conduce
ros2 launch il_control il_control.launch.py mode:=policy \
     policy_type:=il_policy model_path:=models/mlp/best.pt                    # conduce la red
```

Requisito del módulo 1: **Nav2 no puede publicar en `/cmd_vel`**. Hay que remapear la salida del **último** nodo de la cadena de velocidad de Nav2 a `/cmd_vel_expert`. Para saber cuál es:

```bash
ros2 topic info /cmd_vel -v    # el publicador de Nav2 que aparezca (collision_monitor o velocity_smoother)
```

No uséis `/cmd_vel_nav` como nombre: Nav2 ya lo usa internamente para la salida del `controller_server`.

## Decisiones de diseño (para la memoria y la defensa)

* **Nodo selector propio en vez de `twist_mux`.** `twist_mux` elige por prioridad y por quién publicó último. Aquí se necesita exclusión explícita fijada por el experimento, y que la fuente inactiva se ignore aunque siga publicando.
* **La política corre siempre; el mux elige.** Así la red puede ejecutarse en *modo sombra* mientras conduce Nav2, comparando `/cmd_vel_policy` con `/cmd_vel_expert` en el mismo instante. Con eso se mide el MAE online y se observa el *covariate shift* sin arriesgar el robot.
* **Bucle de control con timer a frecuencia fija** (10 Hz para la inferencia, 20 Hz para el mux), no en el callback del LiDAR. El periodo no depende del sensor y el mux sigue publicando ceros aunque las entradas se detengan, que es lo que hace funcionar el watchdog.
* **QoS:** `SensorDataQoS` (best effort) para `/scan`, porque un scan perdido se sustituye por el siguiente y no conviene acumular latencia. Comandos con QoS fiable, profundidad 10. `/il_control/mode` es *transient local*, para que un nodo que arranque tarde conozca el modo actual.
* **Watchdogs:** sin LiDAR reciente (0.5 s) no se puede garantizar la parada por obstáculo, así que el robot se para. Si la fuente activa deja de publicar (0.5 s), también se para y no mantiene el último comando.
* **Proximidad por pasillo, no por sector angular.** Se comprueba un rectángulo del ancho del robot en la dirección de avance: una pared lateral no frena y un obstáculo fino delante sí. Solo se anula la componente lineal y el giro se mantiene, para que la política pueda salir. Hay frenado progresivo entre `slow_distance` y `stop_distance`.
* **Misma seguridad para Nav2 y para la red** (`proximity_check_expert`), para que la comparación sea justa. Las transiciones a `estop_*` se cuentan en `/il_control/status` y servirán como métrica de "casi colisiones".
* **Preprocesado por ángulo y con TF.** Cada rayo va al sector de 4° más cercano en el marco `base_link`, usando el yaw de `rplidar_link` leído de TF (el LiDAR del TB4 puede ir girado respecto a la base). Se toma el **mínimo** por sector: interpolar o promediar difumina obstáculos finos.
* **Objetivo por TF** (`base_link ← frame del goal`) en vez de `/amcl_pose`, que solo se publica cuando AMCL actualiza el filtro. El resultado es matemáticamente igual a `calcular_objetivo_relativo` de `il_data` (hay un test que lo comprueba).

## Contrato con los otros módulos

| Con | Qué | Estado |
|---|---|---|
| `il_policy` (Elena) | `ILPolicy.load(path).predict({'scan', 'goal'}) → (v, w)`. El nº de sectores y el rango máximo se leen del checkpoint | ✅ compatible (test contra `contract.py`) |
| `il_data` (María) | Mismo preprocesado de LiDAR y objetivo al grabar y al inferir | ⚠️ pendiente: su remuestreo es por índice y con interpolación. Propuesta: usar `il_control.observation` como función común |
| `il_expert` (Lucas) | Remapear la salida final de Nav2 a `/cmd_vel_expert` y confirmar el tipo (`TwistStamped` en TB4 Jazzy) | ⚠️ pendiente |
