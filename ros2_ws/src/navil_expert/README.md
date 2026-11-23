
# NavIL Expert - Generación de demostraciones con Nav2

## 1. Descripción

`navil_expert` es un paquete ROS 2 Jazzy que permite generar
demostraciones de navegación autónoma de un TurtleBot 4 en Gazebo,
utilizando Nav2 como controlador experto.

Forma parte del proyecto NavIL de aprendizaje por imitación.

## 2. Funcionalidades

- Envío de objetivos individuales a Nav2.
- Ejecución secuencial de múltiples objetivos.
- Configuración de escenarios mediante YAML.
- Identificación de episodios de navegación.
- Publicación de eventos de inicio, éxito y fallo.
- Comunicación del objetivo con los demás módulos.

## 3. Estructura

```text
navil_expert/
├── config/
│   └── scenarios.yaml
├── navil_expert/
│   ├── __init__.py
│   ├── goal_sender.py
│   └── trajectory_generator.py
├── package.xml
├── setup.py
└── README.md
```

## 4. Requisitos

- ROS 2 Jazzy.
- Gazebo con TurtleBot 4.
- Nav2 y localización AMCL.
- Mapa del entorno.
- Python 3 con PyYAML.

Las instrucciones para recuperar el entorno Docker universitario
se encuentran en `INSTRUCCIONES_ENTORNO.md`, en la raíz del proyecto.

## 5. Compilación

Desde el contenedor Docker:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws

colcon build --packages-select navil_expert --symlink-install

source install/setup.bash
```

## 6. Preparar la navegación

Antes de utilizar este paquete deben estar ejecutándose:

1. Gazebo con TurtleBot 4.
2. AMCL con el mapa del almacén.
3. RViz, con la posición inicial indicada mediante 2D Pose Estimate.
4. Nav2, con sus servidores activos.

Todos deben utilizar tiempo simulado.

## 7. Enviar un objetivo individual

```bash
ros2 run navil_expert goal_sender \
  --x -3.810 \
  --y -0.725 \
  --yaw 0 \
  --ros-args -p use_sim_time:=true
```

Los valores X e Y se expresan en metros en el marco `map`.
El ángulo `yaw` se introduce en grados.

El nodo espera la respuesta de Nav2 e informa del resultado.

## 8. Ejecutar varios objetivos manualmente

```bash
ros2 run navil_expert trajectory_generator \
  --waypoint -3.810 -0.725 0 \
  --waypoint -1.372 2.218 0 \
  --ros-args -p use_sim_time:=true
```

Los objetivos se ejecutan en orden.

Si un objetivo falla, se interrumpe la secuencia.

## 9. Ejecutar escenarios YAML

Los escenarios se definen en `config/scenarios.yaml`.

Ejemplo:

```yaml
escenarios:
  ruta_validada:
    descripcion: "Ruta de dos objetivos"
    waypoints:
      - x: -3.810
        y: -0.725
        yaw: 0.0
      - x: -1.372
        y: 2.218
        yaw: 0.0
```

Para ejecutarlo:

```bash
ros2 run navil_expert trajectory_generator \
  --scenario ruta_validada \
  --ros-args -p use_sim_time:=true
```

Los escenarios incluidos son `ruta_validada` e `ida_vuelta`.
La ruta `ruta_validada` se ha probado correctamente con Nav2.
El escenario `ida_vuelta` requiere validación adicional.

## 10. Interfaces ROS 2

### Acción /navigate_to_pose

Tipo: `nav2_msgs/action/NavigateToPose`

Permite enviar destinos a Nav2 y recibir el resultado de cada
navegación.

### Tópico /navil/goal_pose

Tipo: `geometry_msgs/msg/PoseStamped`

Publica la pose del objetivo cuando Nav2 lo acepta.

Se utiliza un tópico propio para evitar que Nav2 reciba una
segunda solicitud de navegación a través de `/goal_pose`.

### Tópico /navil/episode_event

Tipo: `std_msgs/msg/String`

Contiene un JSON con información del episodio.

Eventos posibles:

- `episode_start`
- `episode_success`
- `episode_failed`

Ejemplo:

```json
{
  "event": "episode_start",
  "episode_id": "33f23ed2_001",
  "scenario": "ruta_validada",
  "waypoint_index": 1,
  "waypoint_total": 2,
  "stamp": 123.45,
  "goal": {
    "x": -3.81,
    "y": -0.725,
    "yaw_deg": 0.0
  }
}
```

El campo `stamp` representa tiempo ROS en segundos.

Cada objetivo se considera un episodio independiente.
El identificador comparte prefijo entre episodios de una
misma ejecución.

El evento de inicio se publica antes de solicitar el objetivo
a Nav2. No representa necesariamente el instante exacto
en que Nav2 empieza a mover el robot.

Para escuchar los eventos:

```bash
ros2 topic echo /navil/episode_event
```

El publicador utiliza QoS reliable y transient local.
Los suscriptores deben estar conectados antes de iniciar
la ruta para asegurar la recepción de los eventos desde
el comienzo. Los eventos no se guardan permanentemente.

## 11. Integración con il_data

El módulo de recopilación de datos deberá:

- Suscribirse a `/navil/goal_pose`.
- Suscribirse a `/navil/episode_event`.
- Asociar los datos grabados al `episode_id`.
- Utilizar los eventos de inicio y finalización para
  delimitar cada demostración.

Esta integración está pendiente de validación conjunta.

## 12. Pruebas realizadas

Se ha verificado:

- Envío de objetivos individuales mediante GoalSender.
- Navegación secuencial a dos destinos.
- Carga y ejecución del escenario `ruta_validada`.
- Publicación y recepción de los cuatro eventos
  correspondientes a dos episodios completados.

Todas estas pruebas se realizaron en el entorno
universitario con ROS 2 Jazzy, Gazebo y TurtleBot 4.
