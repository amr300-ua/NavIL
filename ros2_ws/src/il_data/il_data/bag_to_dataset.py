"""Convierte un rosbag2 de NavIL al formato NPZ del contrato de entrenamiento."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .dataset_io import (
    crear_arrays_dataset,
    crear_informe_calidad,
    guardar_informe_calidad,
    guardar_npz,
)
from .dataset_utils import (
    calcular_objetivo_relativo,
    preprocesar_lidar,
    yaw_desde_cuaternion,
)
from .schema import (
    MuestraNavegacion,
    RANGO_MAXIMO_LIDAR_METROS,
    normalizar_id_episodio,
)


def _stamp_a_nanosegundos(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _seleccionar_topico(
    solicitado: str,
    candidatos: tuple[str, ...],
    tipos_disponibles: dict[str, str],
    conteos: Any,
) -> str:
    if solicitado == "/cmd_vel":
        raise ValueError(
            "/cmd_vel es la salida del mux; no se admite como acción experta"
        )
    if solicitado != "auto":
        return solicitado
    if isinstance(conteos, dict):
        for candidato in candidatos:
            if conteos.get(candidato, 0) > 0:
                return candidato
    for candidato in candidatos:
        if candidato in tipos_disponibles:
            return candidato
    raise ValueError(
        f"No se encontró ningún tópico candidato: {', '.join(candidatos)}"
    )


"""
Este archivo conecta la grabación con el entrenamiento. 
Hasta ahora hemos grabajo ROS2  pero necesitamos recibir muestras esctructuradas.

Prcedimiento:
1.Abre el ROS Bag.
2.Comprueba que contiene los tópicos necesarios.
3.Lee los mensajes.
4.Mantiene guardadas las últimas observaciones de LiDAR, posición y objetivo.
5.Cuando recibe un comando de velocidad, busca las observaciones anteriores más recientes.
6.Comprueba la antigüedad de LiDAR, pose y objetivo.
7.Calcula la distancia y el ángulo al objetivo.
8.Construye una MuestraNavegacion.
9.Guarda las muestras en el NPZ definido por el contrato de entrenamiento.

"""

def convertir(
    ruta_bag: str | Path,
    ruta_salida: str | Path,
    episode_id: int | str | None = None,
    scan_topic: str = "/scan",
    cmd_vel_topic: str = "auto",
    goal_topic: str = "auto",
    antiguedad_maxima: float = 0.5,
    antiguedad_maxima_objetivo: float = 3600.0,
    rango_maximo_lidar: float = RANGO_MAXIMO_LIDAR_METROS,
    desfase_yaw_lidar: float | None = None,
    resultado_episodio: str = "unknown",
    antiguedad_maxima_pose: float = 2.0,
    base_frame: str = "base_link",
    scan_reduction: str = "interp",
    tf_topic: str = "/tf",
    tf_static_topic: str = "/tf_static",
) -> int:
    """Convierte las velocidades usando las observaciones más recientes."""
    if Path(ruta_salida).suffix.lower() != ".npz":
        raise ValueError("El archivo de salida debe tener extensión .npz")
    if not math.isfinite(antiguedad_maxima) or antiguedad_maxima <= 0:
        raise ValueError("La antigüedad máxima debe ser positiva y finita")
    if (
        not math.isfinite(antiguedad_maxima_objetivo)
        or antiguedad_maxima_objetivo <= 0
    ):
        raise ValueError("La antigüedad máxima del objetivo debe ser positiva y finita")
    if not math.isfinite(antiguedad_maxima_pose) or antiguedad_maxima_pose <= 0:
        raise ValueError("La antigüedad máxima de TF debe ser positiva y finita")
    if not math.isfinite(rango_maximo_lidar) or rango_maximo_lidar <= 0:
        raise ValueError("El rango máximo LiDAR debe ser positivo y finito")
    if desfase_yaw_lidar is not None and not math.isfinite(desfase_yaw_lidar):
        raise ValueError("El desfase de orientación LiDAR debe ser finito")
    if scan_reduction not in {"interp", "min"}:
        raise ValueError("scan_reduction debe ser 'interp' o 'min'")
    if not base_frame.strip():
        raise ValueError("base_frame no puede estar vacío")
    resultados_validos = {"success", "failed", "cancelled", "interrupted", "unknown"}
    if resultado_episodio not in resultados_validos:
        raise ValueError(f"Resultado de episodio no válido: {resultado_episodio}")

    ruta_bag = Path(ruta_bag)
    ruta_salida = Path(ruta_salida)
    if ruta_salida.exists():
        raise FileExistsError(f"El dataset de salida ya existe: {ruta_salida}")
    ruta_informe_existente = ruta_salida.with_suffix(".metadata.json")
    if ruta_informe_existente.exists():
        raise FileExistsError(
            f"El informe de salida ya existe: {ruta_informe_existente}"
        )
    ruta_metadatos_episodio = ruta_bag.with_name(
        ruta_bag.name + ".episode.json"
    )
    metadatos_episodio: dict[str, Any] = {}
    id_npz: int | None = None
    id_origen: str | None = None
    if episode_id is not None:
        id_npz, id_origen = normalizar_id_episodio(episode_id)
    if ruta_metadatos_episodio.exists():
        with ruta_metadatos_episodio.open("r", encoding="utf-8") as archivo:
            metadatos_episodio = json.load(archivo)
        id_guardado_npz = metadatos_episodio.get("dataset_episode_id")
        id_guardado_origen = metadatos_episodio.get(
            "source_episode_id", metadatos_episodio.get("episode_id")
        )
        if id_guardado_npz is None:
            id_guardado_npz, id_guardado_origen = normalizar_id_episodio(
                id_guardado_origen
            )
        if (
            id_npz is not None
            and (
                id_npz != id_guardado_npz
                or id_origen not in (str(id_guardado_npz), id_guardado_origen)
            )
        ):
            raise ValueError(
                f"El ID indicado ({id_origen}) no coincide con el del rosbag "
                f"({id_guardado_origen})"
            )
        id_npz = id_guardado_npz
        id_origen = str(id_guardado_origen)
    if id_npz is None or id_origen is None:
        raise ValueError(
            "Falta el ID del episodio. Graba con --episode-id o indica "
            "--episode-id al convertir."
        )
    if isinstance(id_npz, bool) or not isinstance(id_npz, int) or id_npz < 0:
        raise ValueError("Los metadatos del rosbag tienen un episode_id no válido")

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rclpy.time import Time
        from rosidl_runtime_py.utilities import get_message
        from tf2_ros import Buffer, TransformException
    except ImportError as error:
        raise RuntimeError(
            "La conversión de rosbag requiere ROS 2, rosbag2_py, rclpy y tf2_ros."
        ) from error

    lector = rosbag2_py.SequentialReader()
    lector.open(
        rosbag2_py.StorageOptions(uri=str(ruta_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topicos = {entrada.name: entrada.type for entrada in lector.get_all_topics_and_types()}
    conteos = metadatos_episodio.get("message_counts", {})
    cmd_vel_topic = _seleccionar_topico(
        cmd_vel_topic,
        ("/cmd_vel_expert", "/cmd_vel_nav"),
        topicos,
        conteos,
    )
    goal_topic = _seleccionar_topico(
        goal_topic,
        ("/navil/goal_pose", "/goal_pose"),
        topicos,
        conteos,
    )
    requeridos = {
        scan_topic: "sensor_msgs/msg/LaserScan",
        goal_topic: "geometry_msgs/msg/PoseStamped",
        tf_topic: "tf2_msgs/msg/TFMessage",
        tf_static_topic: "tf2_msgs/msg/TFMessage",
    }
    ausentes = sorted(set(requeridos) - topicos.keys())
    if ausentes:
        raise ValueError(f"El rosbag no contiene los tópicos necesarios: {ausentes}")
    tipos_incorrectos = {
        nombre: (topicos[nombre], tipo)
        for nombre, tipo in requeridos.items()
        if topicos[nombre] != tipo
    }
    tipo_velocidad = topicos.get(cmd_vel_topic)
    tipos_velocidad_validos = {
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
    }
    if tipo_velocidad not in tipos_velocidad_validos:
        tipos_incorrectos[cmd_vel_topic] = (
            tipo_velocidad,
            "geometry_msgs/msg/Twist o TwistStamped",
        )
    if tipos_incorrectos:
        raise ValueError(f"Tipos de mensaje inesperados: {tipos_incorrectos}")

    tipos = {nombre: get_message(tipo) for nombre, tipo in requeridos.items()}
    tipos[cmd_vel_topic] = get_message(tipo_velocidad)
    buffer_tf = Buffer()
    ultimo: dict[str, tuple[int, Any]] = {}
    muestras = []
    tiempos_muestra = []
    primera_marca_ns: int | None = None
    ultima_marca_ns: int | None = None
    acciones_totales = 0
    descartadas: dict[str, int] = {}

    def descartar(motivo: str) -> None:
        descartadas[motivo] = descartadas.get(motivo, 0) + 1

    while lector.has_next():
        nombre, datos_serializados, marca_ns = lector.read_next()
        if nombre not in tipos:
            continue
        mensaje = deserialize_message(datos_serializados, tipos[nombre])
        if nombre in {tf_topic, tf_static_topic}:
            insertar_transformacion = (
                buffer_tf.set_transform_static
                if nombre == tf_static_topic
                else buffer_tf.set_transform
            )
            for transformacion in mensaje.transforms:
                insertar_transformacion(transformacion, "il_data_bag")
            continue
        if primera_marca_ns is None:
            primera_marca_ns = marca_ns
        ultima_marca_ns = marca_ns
        ultimo[nombre] = (marca_ns, mensaje)
        if nombre != cmd_vel_topic:
            continue

        acciones_totales += 1

        if scan_topic not in ultimo:
            descartar("missing_scan")
            continue
        if goal_topic not in ultimo:
            descartar("missing_goal")
            continue

        edades = {
            topico: (marca_ns - ultimo[topico][0]) / 1_000_000_000
            for topico in (scan_topic, goal_topic)
        }
        motivo_antiguo = next(
            (
                "future_" + topico.removeprefix("/")
                for topico, edad in edades.items()
                if edad < 0
            ),
            None,
        )
        if motivo_antiguo:
            descartar(motivo_antiguo)
            continue
        if edades[scan_topic] > antiguedad_maxima:
            descartar("stale_scan")
            continue
        if edades[goal_topic] > antiguedad_maxima_objetivo:
            descartar("stale_goal")
            continue

        escaneo = ultimo[scan_topic][1]
        objetivo = ultimo[goal_topic][1]
        marco_objetivo = objetivo.header.frame_id
        if not marco_objetivo:
            descartar("missing_goal_frame")
            continue
        if not escaneo.header.frame_id:
            descartar("missing_scan_frame")
            continue

        try:
            tf_robot = buffer_tf.lookup_transform(
                marco_objetivo, base_frame, Time()
            )
        except TransformException:
            descartar("missing_robot_tf")
            continue
        edad_tf = (
            marca_ns - _stamp_a_nanosegundos(tf_robot.header.stamp)
        ) / 1_000_000_000
        if edad_tf < 0:
            descartar("future_robot_tf")
            continue
        if edad_tf > antiguedad_maxima_pose:
            descartar("stale_robot_tf")
            continue

        try:
            tf_lidar = buffer_tf.lookup_transform(
                base_frame, escaneo.header.frame_id, Time()
            )
        except TransformException:
            descartar("missing_lidar_tf")
            continue

        pose_robot = tf_robot.transform
        pose_objetivo = objetivo.pose
        orientacion = pose_robot.rotation
        try:
            distancia, angulo = calcular_objetivo_relativo(
                pose_robot.translation,
                orientacion,
                pose_objetivo.position,
            )
            giro_lidar = yaw_desde_cuaternion(
                tf_lidar.transform.rotation.x,
                tf_lidar.transform.rotation.y,
                tf_lidar.transform.rotation.z,
                tf_lidar.transform.rotation.w,
            )
            if desfase_yaw_lidar is not None:
                giro_lidar += desfase_yaw_lidar
            scan = preprocesar_lidar(
                escaneo.ranges,
                escaneo.angle_min,
                escaneo.angle_increment,
                escaneo.range_min,
                escaneo.range_max,
                rango_maximo=rango_maximo_lidar,
                desfase_yaw=giro_lidar,
                reduccion=scan_reduction,
            )
            mensaje_velocidad = (
                mensaje.twist if hasattr(mensaje, "twist") else mensaje
            )
            velocidad_lineal = float(mensaje_velocidad.linear.x)
            velocidad_angular = float(mensaje_velocidad.angular.z)
            muestra = MuestraNavegacion(
                marca_temporal=(marca_ns - primera_marca_ns) / 1_000_000_000,
                id_episodio=id_npz,
                lecturas_lidar=scan,
                distancia_objetivo=distancia,
                angulo_objetivo=angulo,
                velocidad_lineal_experta=velocidad_lineal,
                velocidad_angular_experta=velocidad_angular,
            )
        except (ValueError, TypeError, AttributeError, OverflowError):
            descartar("invalid_observation")
            continue
        muestras.append(muestra)
        tiempos_muestra.append(muestra.marca_temporal)

    if not muestras:
        duracion_bag = (
            max(0.0, (ultima_marca_ns - primera_marca_ns) / 1e9)
            if primera_marca_ns is not None and ultima_marca_ns is not None
            else 0.0
        )
        informe_fallido = {
            "episode_id": id_npz,
            "source_episode_id": id_origen,
            "expert_command_topic": cmd_vel_topic,
            "goal_topic": goal_topic,
            "outcome": resultado_episodio,
            "sample_count": 0,
            "action_count": acciones_totales,
            "discarded_count": sum(descartadas.values()),
            "discarded_by_reason": descartadas,
            "valid_percentage": 0.0,
            "duration_seconds": metadatos_episodio.get(
                "duration_seconds", duracion_bag
            ),
            "bag": str(ruta_bag),
            "dataset": str(ruta_salida),
        }
        ruta_informe = guardar_informe_calidad(informe_fallido, ruta_salida)
        raise ValueError(
            "No se generaron muestras válidas. "
            f"Informe de descarte: {ruta_informe}; "
            f"motivos: {descartadas or 'el rosbag no contiene acciones'}"
        )
    cantidad = guardar_npz(muestras, ruta_salida)
    arrays = crear_arrays_dataset(muestras)
    informe = crear_informe_calidad(
        arrays,
        tiempos_muestra,
        acciones_totales,
        descartadas,
        resultado_episodio,
    )
    informe.update(
        {
            "bag": str(ruta_bag),
            "dataset": str(ruta_salida),
            "scan_topic": scan_topic,
            "expert_command_topic": cmd_vel_topic,
            "expert_command_type": tipo_velocidad,
            "pose_source": "tf",
            "base_frame": base_frame,
            "goal_topic": goal_topic,
            "scan_yaw_offset_radians": desfase_yaw_lidar,
            "lidar_tf_yaw_radians": giro_lidar,
            "scan_reduction": scan_reduction,
            "max_scan_age_seconds": antiguedad_maxima,
            "max_tf_pose_age_seconds": antiguedad_maxima_pose,
            "max_goal_age_seconds": antiguedad_maxima_objetivo,
            "recording_time_mode": metadatos_episodio.get(
                "use_sim_time", "unknown"
            ),
            "source_episode_id": id_origen,
        }
    )
    duracion_grabacion = metadatos_episodio.get("duration_seconds")
    if isinstance(duracion_grabacion, (int, float)) and math.isfinite(
        duracion_grabacion
    ):
        informe["duration_seconds"] = float(max(0.0, duracion_grabacion))
    ruta_informe = guardar_informe_calidad(informe, ruta_salida)
    print(f"Dataset guardado: {cantidad} muestras en {ruta_salida}")
    print(f"Informe de episodio guardado en {ruta_informe}")
    print(f"Acción experta: {cmd_vel_topic} | objetivo: {goal_topic}")
    print(
        f"Válidas: {informe['valid_percentage']:.1f}% | "
        f"descartadas: {informe['discarded_count']} | "
        f"duración: {informe['duration_seconds']:.2f} s"
    )
    return cantidad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte rosbag2 al NPZ para Behavioral Cloning."
    )
    parser.add_argument("bag", help="Carpeta del rosbag2")
    parser.add_argument("output", help="Archivo NPZ de salida")
    parser.add_argument(
        "--episode-id",
        help="ID de Lucas (p. ej. 33f23ed2_001) o ID numérico",
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument(
        "--cmd-vel-topic",
        default="auto",
        help=(
            "auto prefiere /cmd_vel_expert y, si no tiene datos, usa "
            "/cmd_vel_nav; nunca usa /cmd_vel"
        ),
    )
    parser.add_argument(
        "--goal-topic",
        default="auto",
        help=(
            "auto prefiere /navil/goal_pose y recurre a /goal_pose solo si "
            "no hay mensajes en el primero"
        ),
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=0.5,
        help="Antigüedad máxima del LiDAR en segundos",
    )
    parser.add_argument(
        "--pose-max-age",
        type=float,
        default=2.0,
        help="Antigüedad máxima de la pose obtenida mediante TF",
    )
    parser.add_argument(
        "--goal-max-age",
        type=float,
        default=3600.0,
        help=(
            "Antigüedad máxima del último evento de objetivo. La meta suele "
            "publicarse una vez y seguir vigente durante el episodio."
        ),
    )
    parser.add_argument(
        "--outcome",
        choices=("success", "failed", "cancelled", "interrupted", "unknown"),
        default="unknown",
        help="Resultado final del episodio para el informe complementario",
    )
    parser.add_argument(
        "--scan-max-range",
        type=float,
        default=RANGO_MAXIMO_LIDAR_METROS,
        help="Distancia en metros usada para lecturas LiDAR sin retorno",
    )
    parser.add_argument(
        "--scan-yaw-offset",
        type=float,
        default=None,
        help="Corrección adicional al yaw de TF del LiDAR (rad), normalmente 0",
    )
    parser.add_argument(
        "--scan-reduction",
        choices=("interp", "min"),
        default="interp",
        help="Reducción LiDAR; debe coincidir con scan_reduction del controlador",
    )
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    args = parser.parse_args()
    convertir(
        args.bag,
        args.output,
        episode_id=args.episode_id,
        scan_topic=args.scan_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        goal_topic=args.goal_topic,
        antiguedad_maxima=args.max_age,
        antiguedad_maxima_pose=args.pose_max_age,
        antiguedad_maxima_objetivo=args.goal_max_age,
        rango_maximo_lidar=args.scan_max_range,
        desfase_yaw_lidar=args.scan_yaw_offset,
        resultado_episodio=args.outcome,
        base_frame=args.base_frame,
        scan_reduction=args.scan_reduction,
        tf_topic=args.tf_topic,
        tf_static_topic=args.tf_static_topic,
    )


if __name__ == "__main__":
    main()
