"""Pruebas del contrato compartido del dataset NavIL."""

import json
import math
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from il_data.dataset_io import (
    cargar_npz,
    crear_arrays_dataset,
    crear_informe_calidad,
    guardar_npz,
)
from il_data.bag_to_dataset import _seleccionar_topico, convertir
from il_data.dataset_utils import (
    calcular_objetivo_relativo,
    laserscan_a_sectores,
    preprocesar_lidar,
    remuestrear_lidar,
)
from il_data.generate_fake_dataset import generar_muestra
from il_data.schema import (
    NUM_LECTURAS_LIDAR,
    RANGO_MAXIMO_LIDAR_METROS,
    MuestraNavegacion,
    normalizar_id_episodio,
)


class PruebasDataset(unittest.TestCase):
    def test_remuestreo_respeta_orientacion_y_angulos_del_contrato(self) -> None:
        cantidad = 360
        angulo_minimo = -math.pi
        incremento = 2 * math.pi / cantidad
        rangos = list(range(cantidad))

        scan = remuestrear_lidar(
            rangos,
            angulo_minimo,
            incremento,
            rango_maximo=500.0,
        )

        self.assertEqual(len(scan), NUM_LECTURAS_LIDAR)
        self.assertAlmostEqual(scan[0], 180.0)
        self.assertAlmostEqual(scan[22], 268.0)
        self.assertAlmostEqual(scan[45], 0.0)
        self.assertAlmostEqual(scan[67], 88.0)

    def test_lecturas_invalidas_y_angulos_sin_cobertura(self) -> None:
        scan = remuestrear_lidar(
            [1.0, math.nan, math.inf, -math.inf, 5.0],
            angulo_minimo=-math.radians(8),
            incremento_angular=math.radians(4),
            rango_maximo=RANGO_MAXIMO_LIDAR_METROS,
        )

        self.assertEqual(scan[0], RANGO_MAXIMO_LIDAR_METROS)
        self.assertEqual(scan[1], 0.0)
        self.assertEqual(scan[89], RANGO_MAXIMO_LIDAR_METROS)
        self.assertTrue(all(math.isfinite(valor) for valor in scan))

    def test_preprocesado_usa_minimo_por_sector_y_range_min(self) -> None:
        ranges = [math.inf] * 360
        ranges[180] = 0.05
        ranges[181] = 1.25

        scan = laserscan_a_sectores(
            ranges,
            angulo_minimo=-math.pi,
            incremento_angular=2 * math.pi / len(ranges),
            rango_minimo=0.15,
            rango_sensor_maximo=12.0,
            rango_maximo=12.0,
        )

        self.assertAlmostEqual(scan[0], 1.25)
        self.assertEqual(scan[45], 12.0)

    def test_reduccion_interp_es_la_predeterminada_y_min_es_opcional(self) -> None:
        ranges = [12.0] * 360
        ranges[181] = 0.4
        args = (
            ranges,
            -math.pi,
            2 * math.pi / len(ranges),
            0.15,
            12.0,
        )

        interpolado = preprocesar_lidar(*args)
        minimo = preprocesar_lidar(*args, reduccion="min")

        self.assertEqual(interpolado[0], 12.0)
        self.assertAlmostEqual(minimo[0], 0.4)

    def test_desfase_yaw_mueve_obstaculo_al_sector_del_frente(self) -> None:
        ranges = [math.inf] * 360
        ranges[90] = 0.4

        scan = laserscan_a_sectores(
            ranges,
            angulo_minimo=-math.pi,
            incremento_angular=2 * math.pi / len(ranges),
            rango_minimo=0.15,
            rango_sensor_maximo=12.0,
            desfase_yaw=math.pi / 2,
        )

        self.assertAlmostEqual(scan[0], 0.4)

    def test_objetivo_positivo_es_hacia_la_izquierda(self) -> None:
        posicion_robot = SimpleNamespace(x=0.0, y=0.0)
        orientacion_robot = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        posicion_objetivo = SimpleNamespace(x=0.0, y=2.0)

        distancia, angulo = calcular_objetivo_relativo(
            posicion_robot, orientacion_robot, posicion_objetivo
        )

        self.assertAlmostEqual(distancia, 2.0)
        self.assertAlmostEqual(angulo, math.pi / 2)

    def test_dataset_npz_tiene_claves_formas_y_tipos_del_contrato(self) -> None:
        muestras = [
            generar_muestra(indice, random.Random(indice), episode_id=7)
            for indice in range(3)
        ]

        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "dataset.npz"
            self.assertEqual(guardar_npz(muestras, ruta), 3)
            datos = cargar_npz(ruta)

        self.assertEqual(set(datos), {"scan", "goal", "v", "w", "episode_id"})
        self.assertEqual(datos["scan"].shape, (3, 90))
        self.assertEqual(datos["goal"].shape, (3, 2))
        self.assertEqual(datos["v"].shape, (3,))
        self.assertEqual(datos["w"].shape, (3,))
        self.assertEqual(datos["episode_id"].shape, (3,))
        for clave in ("scan", "goal", "v", "w"):
            self.assertEqual(datos[clave].dtype, np.dtype(np.float32))
        self.assertEqual(datos["episode_id"].dtype.kind, "i")
        np.testing.assert_array_equal(datos["episode_id"], [7, 7, 7])

    def test_informe_calidad_resume_muestras_validas_y_descartadas(self) -> None:
        muestras = [
            generar_muestra(indice, random.Random(indice), episode_id=5)
            for indice in range(2)
        ]
        datos = crear_arrays_dataset(muestras)

        informe = crear_informe_calidad(
            datos,
            tiempos_muestra=[0.0, 0.4],
            acciones_totales=3,
            muestras_descartadas={"stale_scan": 1},
            resultado_episodio="success",
        )

        self.assertEqual(informe["episode_id"], 5)
        self.assertEqual(informe["sample_count"], 2)
        self.assertEqual(informe["discarded_count"], 1)
        self.assertAlmostEqual(informe["valid_percentage"], 200 / 3)
        self.assertAlmostEqual(informe["duration_seconds"], 0.4)
        self.assertEqual(informe["outcome"], "success")

    def test_converter_elige_topicos_activos_y_conserva_id_de_lucas(self) -> None:
        tipos_por_nombre = {
            "/scan": "sensor_msgs/msg/LaserScan",
            "/cmd_vel_expert": "geometry_msgs/msg/TwistStamped",
            "/cmd_vel_nav": "geometry_msgs/msg/TwistStamped",
            "/navil/goal_pose": "geometry_msgs/msg/PoseStamped",
            "/goal_pose": "geometry_msgs/msg/PoseStamped",
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
        }
        mensajes = []

        def publicar(topico: str, mensaje: object, marca_ns: int) -> None:
            mensajes.append((topico, mensaje, marca_ns))

        def transformacion(
            marco: str,
            hijo: str,
            marca_ns: int,
            x: float = 0.0,
            y: float = 0.0,
            qz: float = 0.0,
            qw: float = 1.0,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                header=SimpleNamespace(
                    frame_id=marco,
                    stamp=SimpleNamespace(
                        sec=marca_ns // 1_000_000_000,
                        nanosec=marca_ns % 1_000_000_000,
                    ),
                ),
                child_frame_id=hijo,
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=x, y=y, z=0.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
                ),
            )

        robot_tf = transformacion(
            "map",
            "base_link",
            1_000_000_000,
            x=1.0,
            y=1.0,
            qz=math.sin(math.pi / 4),
            qw=math.cos(math.pi / 4),
        )
        lidar_tf = transformacion(
            "base_link",
            "laser",
            1_000_000_000,
            qz=math.sin(math.pi / 4),
            qw=math.cos(math.pi / 4),
        )
        scan = SimpleNamespace(
            header=SimpleNamespace(frame_id="laser"),
            ranges=[12.0] * 360,
            angle_min=-math.pi,
            angle_increment=2 * math.pi / 360,
            range_min=0.15,
            range_max=12.0,
        )
        scan.ranges[90] = 0.4
        goal = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=3.0)),
        )
        twist_stamped = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.2),
                angular=SimpleNamespace(z=-0.1),
            )
        )
        publicar(
            "/tf",
            SimpleNamespace(transforms=[robot_tf]),
            1_000_000_000,
        )
        publicar(
            "/tf_static",
            SimpleNamespace(transforms=[lidar_tf]),
            1_000_000_000,
        )
        publicar("/scan", scan, 1_000_000_000)
        publicar("/goal_pose", goal, 1_000_000_000)
        publicar("/cmd_vel_nav", twist_stamped, 1_100_000_000)
        publicar("/scan", scan, 2_000_000_000)
        publicar("/cmd_vel_nav", twist_stamped, 2_100_000_000)
        publicar("/scan", scan, 2_000_000_000)
        publicar("/goal_pose", goal, 2_000_000_000)
        publicar("/cmd_vel_nav", twist_stamped, 2_100_000_000)

        class FakeReader:
            def open(self, *_args: object) -> None:
                pass

            def get_all_topics_and_types(self) -> list:
                return [
                    SimpleNamespace(name=name, type=tipo)
                    for name, tipo in tipos_por_nombre.items()
                ]

            def has_next(self) -> bool:
                return bool(mensajes)

            def read_next(self) -> tuple:
                return mensajes.pop(0)

        rosbag = types.ModuleType("rosbag2_py")
        rosbag.SequentialReader = FakeReader
        rosbag.StorageOptions = lambda **_kwargs: object()
        rosbag.ConverterOptions = lambda **_kwargs: object()
        rclpy = types.ModuleType("rclpy")
        rclpy.__path__ = []
        serialization = types.ModuleType("rclpy.serialization")
        serialization.deserialize_message = lambda message, _message_type: message
        rclpy_time = types.ModuleType("rclpy.time")
        rclpy_time.Time = lambda: object()
        rosidl = types.ModuleType("rosidl_runtime_py")
        rosidl.__path__ = []
        utilities = types.ModuleType("rosidl_runtime_py.utilities")
        utilities.get_message = lambda message_type: message_type

        class FakeTransformException(Exception):
            pass

        class FakeBuffer:
            def __init__(self) -> None:
                self.transforms = {}

            def set_transform(self, transform: object, _authority: str) -> None:
                self.transforms[
                    (transform.header.frame_id, transform.child_frame_id)
                ] = transform

            set_transform_static = set_transform

            def lookup_transform(
                self, target: str, source: str, _time: object
            ) -> object:
                try:
                    return self.transforms[(target, source)]
                except KeyError as error:
                    raise FakeTransformException(str(error)) from error

        tf2_ros = types.ModuleType("tf2_ros")
        tf2_ros.Buffer = FakeBuffer
        tf2_ros.TransformException = FakeTransformException

        with tempfile.TemporaryDirectory() as directorio:
            bag_path = Path(directorio) / "episode"
            output_path = Path(directorio) / "episode.npz"
            bag_path.with_name(bag_path.name + ".episode.json").write_text(
                json.dumps(
                    {
                        "episode_id": "33f23ed2_012",
                        "dataset_episode_id": 12,
                        "message_counts": {
                            "/cmd_vel_expert": 0,
                            "/cmd_vel_nav": 3,
                            "/navil/goal_pose": 0,
                            "/goal_pose": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                sys.modules,
                {
                    "rosbag2_py": rosbag,
                    "rclpy": rclpy,
                    "rclpy.serialization": serialization,
                    "rclpy.time": rclpy_time,
                    "rosidl_runtime_py": rosidl,
                    "rosidl_runtime_py.utilities": utilities,
                    "tf2_ros": tf2_ros,
                },
            ):
                cantidad = convertir(
                    bag_path,
                    output_path,
                    episode_id="33f23ed2_012",
                    antiguedad_maxima_objetivo=0.5,
                    antiguedad_maxima_pose=0.5,
                )
            datos = cargar_npz(output_path)
            informe = json.loads(
                output_path.with_suffix(".metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(cantidad, 1)
        self.assertEqual(datos["episode_id"].tolist(), [12])
        self.assertAlmostEqual(float(datos["scan"][0, 0]), 0.4)
        self.assertAlmostEqual(float(datos["goal"][0, 0]), 2.0)
        self.assertAlmostEqual(float(datos["goal"][0, 1]), 0.0)
        self.assertAlmostEqual(float(datos["v"][0]), 0.2)
        self.assertAlmostEqual(float(datos["w"][0]), -0.1)
        self.assertEqual(
            informe["discarded_by_reason"],
            {"stale_goal": 1, "stale_robot_tf": 1},
        )
        self.assertEqual(informe["action_count"], 3)
        self.assertEqual(informe["outcome"], "unknown")
        self.assertEqual(informe["source_episode_id"], "33f23ed2_012")
        self.assertEqual(informe["expert_command_topic"], "/cmd_vel_nav")
        self.assertEqual(informe["goal_topic"], "/goal_pose")
        self.assertAlmostEqual(informe["lidar_tf_yaw_radians"], math.pi / 2)

    def test_id_de_lucas_se_traduce_a_entero_y_conserva_su_texto(self) -> None:
        self.assertEqual(
            normalizar_id_episodio("33f23ed2_001"),
            (1, "33f23ed2_001"),
        )

    def test_no_permite_grabar_velocidad_final_del_mux_como_experta(self) -> None:
        with self.assertRaisesRegex(ValueError, "salida del mux"):
            _seleccionar_topico(
                "/cmd_vel",
                ("/cmd_vel_expert", "/cmd_vel_nav"),
                {},
                {},
            )

    def test_rechaza_dataset_npz_con_forma_incorrecta(self) -> None:
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "dataset.npz"
            np.savez(
                ruta,
                scan=np.zeros((1, 89), dtype=np.float32),
                goal=np.zeros((1, 2), dtype=np.float32),
                v=np.zeros(1, dtype=np.float32),
                w=np.zeros(1, dtype=np.float32),
                episode_id=np.zeros(1, dtype=np.int32),
            )

            with self.assertRaisesRegex(ValueError, "forma"):
                cargar_npz(ruta)

    def test_esquema_rechaza_id_de_episodio_no_entero(self) -> None:
        with self.assertRaisesRegex(ValueError, "entero no negativo"):
            MuestraNavegacion(
                marca_temporal=0.0,
                id_episodio="episodio_1",
                lecturas_lidar=[1.0] * NUM_LECTURAS_LIDAR,
                distancia_objetivo=1.0,
                angulo_objetivo=0.0,
                velocidad_lineal_experta=0.1,
                velocidad_angular_experta=0.0,
            )


if __name__ == "__main__":
    unittest.main()
