# Recuperación y arranque del entorno — NavIL (integrante 1)

> Práctica II de Robótica Inteligente. Módulo `navil_expert`: simulación de TurtleBot 4, navegación con Nav2 y generación de demostraciones expertas.
>
> **Importante:** los ordenadores de la universidad pueden borrar los cambios locales al apagarse. **El repositorio GitHub es la copia de seguridad del código y del mapa, NO de la imagen ni del contenedor Docker.**

## 1. Qué está guardado y qué no

- **En GitHub**, rama `feature/il-expert`: paquete `ros2_ws/src/navil_expert/`, mapa `ros2_ws/maps/warehouse.yaml` y `warehouse.pgm`, y código que se haya confirmado y subido.
- **No queda guardado en GitHub**: contenedor e imagen Docker, sesiones de Gazebo/Nav2/RViz, cambios realizados *dentro* del contenedor fuera del volumen compartido, instalaciones puntuales en el contenedor y archivos sin commit.
- La imagen utilizada en la universidad se llamaba **`robotica_inteligente_grado:latest`**, con ROS 2 Jazzy y los paquetes de simulación. Si la universidad también la elimina, habrá que recuperarla a través del entorno/material de la asignatura; **este repositorio por sí solo no la reconstruye necesariamente**. No utilizar el `docker/compose.yaml` del repositorio sin revisarlo: se creó inicialmente para otro entorno (WSL).

## 2. Recuperar el proyecto en Ubuntu (FUERA de Docker)

En una terminal de Ubuntu:

```bash
# Si la carpeta NavIL aún no existe:
git clone --branch feature/il-expert https://github.com/amr300-ua/NavIL.git ~/NavIL

# Si ya existe, no la clones de nuevo; actualízala:
cd ~/NavIL
git switch feature/il-expert
git pull --ff-only

# Verificar la rama y la presencia del mapa:
git status -sb
ls -lh ros2_ws/maps/warehouse.yaml ros2_ws/maps/warehouse.pgm
```

> **Nota:** el comando `git clone` se utiliza solo cuando no existe `~/NavIL`. Si ya existe, ejecutar únicamente el bloque de actualización.

## 3. Comprobar Docker y crear el contenedor (FUERA de Docker)

```bash
docker --version
docker image ls robotica_inteligente_grado
# ¿Existe un contenedor con el nombre esperado?
docker ps -a --filter name=robotica_inteligente_grado
```

**Caso A: el contenedor existe**. Arrancarlo y entrar:

```bash
docker start robotica_inteligente_grado
docker exec -it --user developer robotica_inteligente_grado bash
```

**Caso B: el contenedor ya no existe, pero la imagen `robotica_inteligente_grado:latest` sí**. Crear uno nuevo **desde una terminal Ubuntu**:

```bash
docker run -it \
  --user developer \
  --network host --ipc host --pid host \
  --env DISPLAY="$DISPLAY" \
  --env QT_X11_NO_MITSHM=1 \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env __NV_PRIME_RENDER_OFFLOAD=1 \
  --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
  --env __GL_THREADED_OPTIMIZATIONS=0 \
  --env ROS_DOMAIN_ID=2 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$HOME/NavIL/ros2_ws:/home/developer/ros2_ws:rw" \
  --device /dev/dri \
  --gpus all \
  --name robotica_inteligente_grado \
  robotica_inteligente_grado:latest
```

Este comando deja una terminal abierta **dentro de Docker**. Para abrir más terminales independientes, ejecutar desde otras terminales Ubuntu:

```bash
docker exec -it --user developer robotica_inteligente_grado bash
```

**Caso C: falta también la imagen**. Recuperar la imagen oficial de la asignatura antes de seguir. No hay garantía de que la imagen se conserve al reiniciar un equipo de la universidad.

### Volumen compartido

| En Ubuntu | Dentro de Docker |
| --- | --- |
| `~/NavIL/ros2_ws` | `/home/developer/ros2_ws` |

Editar código en VS Code **sobre `~/NavIL` en Ubuntu**. Compilar y ejecutar ROS 2 **dentro de Docker**. Los archivos que estén en la carpeta compartida pueden guardarse en Git; los cambios fuera de ella podrían perderse.

## 4. Preparar el workspace ROS 2 (DENTRO de Docker)

Después de entrar en el contenedor:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=2
cd ~/ros2_ws
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash

# Comprobaciones:
ros2 pkg list | grep navil_expert
ros2 pkg executables navil_expert
```


**Estado actual:** el paquete `navil_expert` contiene dos ejecutables funcionales:

- `goal_sender`: envía un objetivo individual a Nav2 y espera su resultado.
- `trajectory_generator`: ejecuta varios objetivos consecutivos, permite cargar escenarios YAML y publica eventos de episodios.

Ambos ejecutables han sido probados en el entorno universitario con TurtleBot 4 y Nav2.


En **cada nueva terminal Docker** utilizada para lanzar procesos, ejecutar primero:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=2
```

Si el código cambió después de la compilación, volver a ejecutar `colcon build --packages-select navil_expert --symlink-install` y cargar de nuevo `source ~/ros2_ws/install/setup.bash`.

## 5. Poner en marcha la navegación (4 terminales DENTRO de Docker)

Abrir cinco terminales independientes con `docker exec -it --user developer robotica_inteligente_grado bash` desde Ubuntu, y preparar el entorno con las tres líneas `source / export` anteriores en cada una.

### Terminal Docker 1 — Gazebo y TurtleBot 4

```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py use_sim_time:=true
```

Esperar a que Gazebo cargue el mundo `warehouse` y el robot. No cerrar esta terminal mientras se use la simulación.


### Comprobar el reloj y los sensores

Después de iniciar Gazebo, comprobar que el puente de reloj
incluido en el lanzamiento funciona correctamente.

Desde otra terminal Docker:

```bash
timeout 6 ros2 topic hz /clock
timeout 6 ros2 topic hz /odom
timeout 6 ros2 topic hz /scan
```

En las pruebas realizadas se observaron frecuencias aproximadas de:

- `/clock`: 266 Hz.
- `/odom`: 50 Hz.
- `/scan`: 50 Hz.

No iniciar un segundo puente de `/clock` por defecto.

Si dejan de recibirse mensajes o aparecen errores de TF,
comprobar primero el estado de Gazebo, los publicadores de
`/clock` y la coherencia del tiempo simulado.

Evitar reiniciar Gazebo mientras AMCL y Nav2 continúan
utilizando transformaciones de una simulación anterior.
Si es necesario reiniciar completamente, detener también
los nodos de navegación y localización.


### Terminal Docker 2 — Localización sobre el mapa guardado

```bash
ros2 launch turtlebot4_navigation localization.launch.py \
  map:=/home/developer/ros2_ws/maps/warehouse.yaml \
  use_sim_time:=true
```

Se usa el mapa que ya está en GitHub; **no hace falta repetir SLAM**.

### Terminal Docker 3 — Nav2

```bash
ros2 launch turtlebot4_navigation nav2.launch.py use_sim_time:=true
```

### Terminal Docker 4 — RViz

```bash
ros2 launch turtlebot4_viz view_navigation.launch.py use_sim_time:=true
```

En RViz:

1. Comprobar que aparece el mapa del almacén.
2. Usar **2D Pose Estimate** para indicar la posición y orientación inicial aproximadas del TurtleBot 4 en el mapa (AMCL).
3. Esperar a que la localización sea coherente y usar **Nav2 Goal** para elegir un destino libre.
4. Verificar que el robot navega autónomamente; **esto ya funcionó en una sesión anterior**.

## 6. Comprobaciones útiles y resolución de incidencias

En otra terminal **dentro de Docker** (con las líneas `source / export`):

```bash
ros2 action list -t | grep navigate_to_pose
ros2 topic list | grep -E '^/(clock|scan|odom|amcl_pose)$'
ros2 pkg prefix navil_expert
```

- **No hay `/clock` o `/odom`:** comprobar que Gazebo está corriendo y que la terminal del puente de `/world/warehouse/clock` no se ha cerrado.
- **El mapa no aparece en RViz:** comprobar la ruta `/home/developer/ros2_ws/maps/warehouse.yaml`, que `warehouse.pgm` está junto a ella y que la localización se ha iniciado. Anteriormente hubo que recuperar el `map_server` tras una caída de Gazebo.
- **Nav2 no recibe objetivos o no navega:** comprobar que sus nodos han terminado de arrancar, que hay transformaciones TF válidas y que se ha realizado `2D Pose Estimate`.
- **Gazebo se cierra inesperadamente:** en las pruebas anteriores hubo una caída de OGRE con *segmentation fault*. Volver a iniciarlo y comprobar reloj, mapa y TF antes de seguir.
- **`ros2 run navil_expert goal_sender` no encuentra el ejecutable:** reconstruir con `colcon build --packages-select navil_expert --symlink-install` y ejecutar `source ~/ros2_ws/install/setup.bash`.
- **`--gpus all` falla:** verificar los controladores NVIDIA y el soporte de Docker para GPU en ese equipo; el código en Git no instala esos requisitos.

## 7. Guardar absolutamente todo el código antes de apagar (FUERA de Docker)

Guardar en VS Code (`Ctrl+S`), abrir una terminal de **Ubuntu** y revisar:

```bash
cd ~/NavIL
git status -sb
git diff --stat
git ls-files ros2_ws/maps/
```

Si hay cambios válidos en archivos del proyecto, añadirlos expresamente, crear un commit descriptivo en español y subir a **nuestra rama**:

```bash
# EJEMPLO: sustituir la ruta por los archivos que de verdad hayan cambiado.
git add INSTRUCCIONES_ENTORNO.md
git commit -m "Documentar recuperacion del entorno ROS 2 y Nav2"
git -c credential.helper= push origin feature/il-expert
git status -sb
```

El último comando debe indicar la rama sincronizada, **sin archivos modificados o sin seguimiento**. No subir tokens, contraseñas, credenciales, carpetas `build/`, `install/`, `log/` ni archivos binarios temporales. Evitar guardar las credenciales de GitHub en el equipo compartido.

Cuando ya esté todo en GitHub, detener los procesos ROS 2 con `Ctrl+C` en sus terminales y, desde Ubuntu, detener Docker con:

```bash
docker stop robotica_inteligente_grado
```

No ejecutar `docker rm` salvo que se pretenda eliminar el contenedor.

#
## 8. Estado actual del módulo navil_expert

### Funcionalidades completadas

- Simulación de TurtleBot 4 mediante Gazebo.
- Mapa del almacén generado y guardado.
- Localización mediante AMCL.
- Navegación autónoma con Nav2.
- Envío de objetivos individuales mediante `goal_sender`.
- Ejecución secuencial de objetivos mediante `trajectory_generator`.
- Escenarios configurables en `config/scenarios.yaml`.
- Identificación individual de episodios.
- Publicación de eventos de inicio, éxito y fallo.

### Pruebas realizadas

Se ha comprobado que:

- Nav2 acepta y completa objetivos enviados desde Python.
- El generador ejecuta dos objetivos consecutivamente.
- El escenario YAML `ruta_validada` funciona correctamente.
- Se reciben los cuatro eventos correspondientes a dos episodios completados.
- Cada episodio conserva su identificador entre el inicio y el final.

### Interfaces para los otros integrantes

- `/navigate_to_pose`: acción utilizada para controlar Nav2.
- `/navil/goal_pose`: publica el objetivo aceptado.
- `/navil/episode_event`: publica eventos de episodios en formato JSON.

Los demás módulos deben utilizar `/navil/goal_pose` en lugar de
`/goal_pose` para evitar duplicar solicitudes de navegación a Nav2.

### Trabajo pendiente

- Integrar los eventos y objetivos con el grabador `il_data`.
- Validar otros escenarios y situaciones de fallo.
- Realizar pruebas conjuntas con los módulos de aprendizaje y control.

La implementación principal del módulo experto está completada.

Consultar `ros2_ws/src/navil_expert/README.md` para conocer los
comandos de ejecución y el formato de los mensajes.
