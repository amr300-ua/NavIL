"""Permite ejecutar `pytest src/il_control/test` sin haber hecho colcon build."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
