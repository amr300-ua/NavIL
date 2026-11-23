"""Pruebas del contrato compartido del dataset NavIL."""

import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from il_data.dataset_io import cargar_npz, guardar_npz
from il_data.dataset_utils import calcular_objetivo_relativo, remuestrear_lidar
from il_data.generate_fake_dataset import generar_muestra
from il_data.schema import (
    NUM_LECTURAS_LIDAR,
    RANGO_MAXIMO_LIDAR_METROS,
    MuestraNavegacion,
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
