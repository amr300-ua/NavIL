"""Pruebas del formato y transformaciones del dataset NavIL."""

import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from il_data.dataset_io import cargar_jsonl, guardar_jsonl
from il_data.dataset_utils import calcular_objetivo_relativo, remuestrear_lidar
from il_data.generate_fake_dataset import generar_muestra
from il_data.schema import NUM_LECTURAS_LIDAR

"""
Pruebas del formato y transformaciones del dataset NavIL.
Nos permite comprobar que las funciones anteriores trabajan correctamente.
"""


class PruebasDataset(unittest.TestCase):
    def test_remuestreo_lidar_normaliza_longitud_y_valores(self) -> None:
        lecturas = remuestrear_lidar(
            [0.0, 1.0, math.nan, math.inf, -math.inf, 9.0],
            rango_minimo=0.1,
            rango_maximo=8.0,
        )

        self.assertEqual(len(lecturas), NUM_LECTURAS_LIDAR)
        self.assertTrue(all(math.isfinite(valor) for valor in lecturas))
        self.assertTrue(all(0.1 <= valor <= 8.0 for valor in lecturas))
        self.assertAlmostEqual(lecturas[0], 0.1)
        self.assertAlmostEqual(lecturas[-1], 8.0)

    def test_objetivo_se_expresa_en_marco_del_robot(self) -> None:
        posicion_robot = SimpleNamespace(x=0.0, y=0.0)
        orientacion_robot = SimpleNamespace(
            x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5)
        )
        posicion_objetivo = SimpleNamespace(x=0.0, y=2.0)

        distancia, angulo = calcular_objetivo_relativo(
            posicion_robot, orientacion_robot, posicion_objetivo
        )

        self.assertAlmostEqual(distancia, 2.0)
        self.assertAlmostEqual(angulo, 0.0)

    def test_dataset_ficticio_se_guarda_y_se_recarga(self) -> None:
        primera = generar_muestra(0, random.Random(42))
        segunda = generar_muestra(0, random.Random(42))
        self.assertEqual(primera, segunda)

        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "dataset.jsonl"
            self.assertEqual(guardar_jsonl([primera], ruta), 1)
            recargadas = cargar_jsonl(ruta)

        self.assertEqual(recargadas, [primera])

    def test_dataset_invalido_indica_numero_de_linea(self) -> None:
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "dataset.jsonl"
            ruta.write_text('{"marca_temporal": NaN}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "línea 1"):
                cargar_jsonl(ruta)


if __name__ == "__main__":
    unittest.main()
