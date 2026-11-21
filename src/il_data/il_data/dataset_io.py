"""Lectura y escritura del formato JSONL del dataset NavIL."""

import json
from pathlib import Path
from typing import Iterable

from .schema import MuestraNavegacion

""" 
Este archivo permite almacenar y recuperar las muestras del dataset.
"""

"""
guardar_jsonl escribe un dataset en un archivo JSONL,
escribe las muestras 
y cargar_jsonl las recupera, conviertiendolas otra vez en objetos
MuestraNavegacion.
"""
def guardar_jsonl(
    muestras: Iterable[MuestraNavegacion], ruta: str | Path
) -> int:
    """Escribe una muestra JSON por línea y devuelve el número escrito."""
    cantidad = 0
    ruta_archivo = Path(ruta)
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    with ruta_archivo.open("w", encoding="utf-8") as archivo:
        for muestra in muestras:
            archivo.write(
                json.dumps(muestra.a_diccionario(), allow_nan=False) + "\n"
            )
            cantidad += 1
    return cantidad


def cargar_jsonl(ruta: str | Path) -> list[MuestraNavegacion]:
    """Lee y valida un dataset JSONL, indicando la línea si hay errores."""
    muestras = []
    with Path(ruta).open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                datos = json.loads(linea)
                if not isinstance(datos, dict):
                    raise ValueError("cada línea debe contener un objeto JSON")
                muestras.append(MuestraNavegacion.desde_diccionario(datos))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Dataset no válido en la línea {numero_linea}: {error}"
                ) from error
    return muestras
