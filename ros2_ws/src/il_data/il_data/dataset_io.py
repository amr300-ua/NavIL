"""Lectura, escritura y validación del formato NPZ del contrato NavIL."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import MuestraNavegacion, NUM_LECTURAS_LIDAR

CLAVES_DATASET = ("scan", "goal", "v", "w", "episode_id")


def crear_arrays_dataset(
    muestras: Sequence[MuestraNavegacion],
) -> dict[str, np.ndarray]:
    """Convierte muestras NavIL a las cinco matrices del contrato compartido."""
    if not muestras:
        raise ValueError("No se puede guardar un dataset vacío")

    arrays = {
        "scan": np.asarray(
            [muestra.lecturas_lidar for muestra in muestras], dtype=np.float32
        ),
        "goal": np.asarray(
            [
                [muestra.distancia_objetivo, muestra.angulo_objetivo]
                for muestra in muestras
            ],
            dtype=np.float32,
        ),
        "v": np.asarray(
            [muestra.velocidad_lineal_experta for muestra in muestras],
            dtype=np.float32,
        ),
        "w": np.asarray(
            [muestra.velocidad_angular_experta for muestra in muestras],
            dtype=np.float32,
        ),
        "episode_id": np.asarray(
            [muestra.id_episodio for muestra in muestras], dtype=np.int64
        ),
    }
    validar_arrays_dataset(arrays)
    return arrays


def validar_arrays_dataset(
    datos: Mapping[str, np.ndarray],
) -> int:
    """Valida las claves, formas y tipos que consume el entrenamiento."""
    faltantes = [clave for clave in CLAVES_DATASET if clave not in datos]
    if faltantes:
        raise KeyError(f"Faltan claves en el dataset: {faltantes}")

    cantidad = len(datos["v"])
    formas_esperadas = {
        "scan": (cantidad, NUM_LECTURAS_LIDAR),
        "goal": (cantidad, 2),
        "v": (cantidad,),
        "w": (cantidad,),
        "episode_id": (cantidad,),
    }
    if cantidad == 0:
        raise ValueError("Dataset vacío")
    for clave, forma in formas_esperadas.items():
        if datos[clave].shape != forma:
            raise ValueError(
                f"{clave}: forma {datos[clave].shape}, esperada {forma}"
            )

    for clave in ("scan", "goal", "v", "w"):
        if datos[clave].dtype != np.dtype(np.float32):
            raise ValueError(f"{clave} debe tener tipo float32")
        if not np.all(np.isfinite(datos[clave])):
            raise ValueError(f"{clave} contiene NaN o infinito")
    if np.any(datos["scan"] < 0):
        raise ValueError("scan contiene distancias negativas")
    if np.any(datos["goal"][:, 0] < 0):
        raise ValueError("goal contiene distancias negativas")
    if np.any(np.abs(datos["goal"][:, 1]) > np.pi):
        raise ValueError("goal contiene ángulos fuera del rango [-pi, pi]")
    if datos["episode_id"].dtype.kind not in "iu":
        raise ValueError("episode_id debe tener un tipo entero")
    if np.any(datos["episode_id"] < 0):
        raise ValueError("episode_id debe contener enteros no negativos")
    return cantidad


def guardar_npz(
    muestras: Sequence[MuestraNavegacion], ruta: str | Path
) -> int:
    """Guarda el dataset comprimido con las claves del contrato de ML."""
    ruta_archivo = Path(ruta)
    if ruta_archivo.suffix.lower() != ".npz":
        raise ValueError("El archivo de salida debe tener extensión .npz")
    arrays = crear_arrays_dataset(muestras)
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    with ruta_archivo.open("wb") as archivo:
        np.savez_compressed(archivo, **arrays)
    return len(muestras)


def cargar_npz(ruta: str | Path) -> dict[str, np.ndarray]:
    """Carga y valida un dataset NPZ sin permitir arrays pickle."""
    with np.load(ruta, allow_pickle=False) as archivo:
        datos = {clave: archivo[clave] for clave in archivo.files}
    validar_arrays_dataset(datos)
    return datos


def crear_informe_calidad(
    datos: Mapping[str, np.ndarray],
    tiempos_muestra: Sequence[float],
    acciones_totales: int,
    muestras_descartadas: Mapping[str, int],
    resultado_episodio: str,
) -> dict[str, Any]:
    """Calcula estadísticas básicas de un episodio ya validado."""
    cantidad = validar_arrays_dataset(datos)
    if np.unique(datos["episode_id"]).size != 1:
        raise ValueError("El informe individual requiere exactamente un episodio")
    if len(tiempos_muestra) != cantidad:
        raise ValueError("Debe proporcionarse un tiempo por cada muestra válida")
    if acciones_totales < cantidad:
        raise ValueError("El total de acciones no puede ser menor que las muestras")
    tiempos = np.asarray(tiempos_muestra, dtype=np.float64)
    if not np.all(np.isfinite(tiempos)):
        raise ValueError("Los tiempos de muestra deben ser finitos")
    if tiempos.size > 1 and np.any(np.diff(tiempos) < 0):
        raise ValueError("Los tiempos de muestra deben estar ordenados")
    if any(numero < 0 for numero in muestras_descartadas.values()):
        raise ValueError("Los contadores de muestras descartadas no pueden ser negativos")

    def estadisticas(valores: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(valores)),
            "mean": float(np.mean(valores)),
            "max": float(np.max(valores)),
        }

    total_descartadas = sum(muestras_descartadas.values())
    return {
        "episode_id": int(datos["episode_id"][0]),
        "outcome": resultado_episodio,
        "sample_count": cantidad,
        "action_count": acciones_totales,
        "discarded_count": total_descartadas,
        "discarded_by_reason": dict(muestras_descartadas),
        "valid_percentage": (
            100.0 * cantidad / acciones_totales if acciones_totales else 0.0
        ),
        "duration_seconds": (
            float(tiempos[-1] - tiempos[0]) if tiempos.size > 1 else 0.0
        ),
        "v_meters_per_second": estadisticas(datos["v"]),
        "w_radians_per_second": estadisticas(datos["w"]),
        "goal_distance_meters": estadisticas(datos["goal"][:, 0]),
        "goal_angle_radians": estadisticas(datos["goal"][:, 1]),
    }


def guardar_informe_calidad(
    informe: Mapping[str, Any], ruta_npz: str | Path
) -> Path:
    """Escribe el informe JSON junto al NPZ, sin alterar el contrato NPZ."""
    ruta_informe = Path(ruta_npz).with_suffix(".metadata.json")
    ruta_informe.parent.mkdir(parents=True, exist_ok=True)
    with ruta_informe.open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, indent=2, allow_nan=False)
        archivo.write("\n")
    return ruta_informe
