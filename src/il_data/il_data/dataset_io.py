"""Lectura, escritura y validación del formato NPZ del contrato NavIL."""

from collections.abc import Mapping, Sequence
from pathlib import Path

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
