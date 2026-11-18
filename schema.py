
"""
Define el esquema de datos para el aprendizaje por imitación
en navegación (NavIL).

Cada MuestraNavegacion representa una observación del robot
junto con la acción correspondiente proporcionada por Nav2.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any


# Configuración del esquema
VERSION_ESQUEMA = "1.0"

# Número inicial de lecturas LiDAR que utilizará el modelo.
# Este valor podrá modificarse al comprobar la configuración
# real del sensor.
NUM_LECTURAS_LIDAR = 90


@dataclass
class MuestraNavegacion:
    """
    Representa una pareja observación-acción para el aprendizaje
    por imitación.

    Las lecturas LiDAR deben estar previamente procesadas:
    - Longitud fija (NUM_LECTURAS_LIDAR).
    - Distancias expresadas en metros.
    - Sin valores NaN ni infinitos.
    """

    marca_temporal: float
    id_episodio: str
    lecturas_lidar: list[float]

    distancia_objetivo: float
    angulo_objetivo: float

    velocidad_lineal_experta: float
    velocidad_angular_experta: float

    version_esquema: str = VERSION_ESQUEMA

    def __post_init__(self) -> None:
        """Comprueba que los datos de la muestra sean válidos."""

        if not math.isfinite(self.marca_temporal) or self.marca_temporal < 0:
            raise ValueError(
                "La marca temporal debe ser un número válido y no negativo"
            )

        if not self.id_episodio.strip():
            raise ValueError("El identificador del episodio no puede estar vacío")

        if len(self.lecturas_lidar) != NUM_LECTURAS_LIDAR:
            raise ValueError(
                f"Las lecturas LiDAR deben contener exactamente "
                f"{NUM_LECTURAS_LIDAR} valores, pero se han recibido "
                f"{len(self.lecturas_lidar)}"
            )

        for indice, distancia in enumerate(self.lecturas_lidar):
            if not math.isfinite(distancia) or distancia < 0:
                raise ValueError(
                    f"Lectura LiDAR no válida en la posición {indice}: "
                    f"{distancia}"
                )

        if not math.isfinite(self.distancia_objetivo) or self.distancia_objetivo < 0:
            raise ValueError(
                "La distancia al objetivo debe ser válida y no negativa"
            )

        if not math.isfinite(self.angulo_objetivo):
            raise ValueError("El ángulo al objetivo debe ser un número válido")

        if not -math.pi <= self.angulo_objetivo <= math.pi:
            raise ValueError("El ángulo al objetivo debe estar entre -pi y pi")

        if not math.isfinite(self.velocidad_lineal_experta):
            raise ValueError("La velocidad lineal experta no es válida")

        if not math.isfinite(self.velocidad_angular_experta):
            raise ValueError("La velocidad angular experta no es válida")

    def a_diccionario(self) -> dict[str, Any]:
        """Convierte la muestra en un diccionario."""
        return asdict(self)

    @classmethod
    def desde_diccionario(cls, datos: dict[str, Any]) -> "MuestraNavegacion":
        """Crea una muestra a partir de un diccionario."""

        return cls(
            marca_temporal=float(datos["marca_temporal"]),
            id_episodio=str(datos["id_episodio"]),
            lecturas_lidar=[
                float(valor) for valor in datos["lecturas_lidar"]
            ],
            distancia_objetivo=float(datos["distancia_objetivo"]),
            angulo_objetivo=float(datos["angulo_objetivo"]),
            velocidad_lineal_experta=float(
                datos["velocidad_lineal_experta"]
            ),
            velocidad_angular_experta=float(
                datos["velocidad_angular_experta"]
            ),
            version_esquema=str(
                datos.get("version_esquema", VERSION_ESQUEMA)
            ),
        )
