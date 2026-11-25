# il_policy — Módulo 3: política de Behavioral Cloning

Rama: `feature/il-policy` · Estado: **ciclo 1 (modelo con datos ficticios)**

Qué hace este paquete: entrena una red MLP que imita al experto (Nav2). La red recibe el LiDAR y el objetivo relativo y predice `(v, w)`. Ofrece una interfaz `predict(observation)` que usará el nodo de inferencia del módulo 4.

```
90 LiDAR + [dist, ángulo]  →  Dense 128 → Dense 128 → Dense 64 → tanh  →  (v, w)
```

## Estructura

| Fichero | Contenido |
|---|---|
| `il_policy/contract.py` | **Contrato de datos** (tamaños, convención de ángulos, normalización, límites, formato del dataset). No depende de PyTorch. |
| `il_policy/model.py` | `MLPPolicy` (92→128→128→64→2) |
| `il_policy/synthetic.py` | Generador de datos ficticios: mundo 2D + LiDAR simulado + experto reactivo |
| `il_policy/dataset.py` | Carga de `.npz` y separación train/val/test **por episodios** |
| `il_policy/train.py` | Entrenamiento (AdamW, MSE, early stopping) → `best.pt`, `metrics.json`, `history.csv` |
| `il_policy/policy.py` | `ILPolicy.load(...)`, `predict(obs) → (v, w)` y `predict_batch` |
| `il_policy/evaluate.py` | Métricas offline (MAE, RMSE, R², baseline) y latencia |
| `config/policy_mlp.yaml` | Hiperparámetros y límites por defecto |
| `test/` | 15 tests con pytest (contrato, modelo, inferencia, entrenamiento de extremo a extremo) |

## Instalación (dentro del contenedor)

El Dockerfile no incluye PyTorch. Basta con la versión CPU. Añadid esto al Dockerfile (Ubuntu 24.04 exige `--break-system-packages`):

```dockerfile
RUN pip3 install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
cd ~/ros2_ws
colcon build --packages-select il_policy --symlink-install
source install/setup.bash
```

## Uso — ciclo 1

```bash
ros2 run il_policy make_dummy_dataset --out data/dummy.npz --episodes 200
ros2 run il_policy train    --data data/dummy.npz --out models/mlp_dummy
ros2 run il_policy evaluate --model models/mlp_dummy/best.pt --data data/dummy.npz
ros2 run il_policy predict_demo --model models/mlp_dummy/best.pt
```

O todo de una vez: `./scripts/train_policy_dummy.sh` (sin compilar: `USE_SOURCE=1 ./scripts/train_policy_dummy.sh`).

Tests: `colcon test --packages-select il_policy` o `python3 -m pytest src/il_policy/test`.

Opciones útiles de `train`: `--epochs`, `--lr`, `--batch-size`, `--hidden 256 128 64`, `--dropout 0.1`, `--device cuda`, `--config otro.yaml`. `--data` acepta varios `.npz` o un directorio.

### Resultado de referencia (200 episodios ficticios, CPU, ~25 s)

| | MAE modelo | MAE baseline (media) | R² |
|---|---|---|---|
| v (m/s) | 0.022 | 0.064 | 0.79 |
| w (rad/s) | 0.122 | 0.142 | 0.38 |

Latencia de `predict()`: ~0.1 ms, de sobra para un bucle de control a 10–20 Hz.

## Rama il-control

```python
from il_policy.policy import ILPolicy

policy = ILPolicy.load('models/mlp_dummy/best.pt')   # o ILPolicy.untrained() sin modelo
v, w = policy.predict({'scan': scan_90, 'goal': [dist, ang]})
```

* Acepta listas o arrays de numpy. `inf`/`NaN` en el LiDAR se tratan como "libre".
* La salida ya va recortada a `v_max = 0.30 m/s` y `w_max = 1.5 rad/s` y nunca es NaN. Aun así, **la seguridad (parada por obstáculo cercano, timeout de sensores, selector experto/modelo) es responsabilidad del nodo de control**, no del modelo.
* Lanza `ValueError`/`KeyError` si la observación no cumple el contrato. Capturadlo y publicad velocidad 0.

## Rama il-data

El entrenamiento espera `.npz` con estas claves:

| clave | forma | tipo |
|---|---|---|
| `scan` | (N, 90) | float32, metros |
| `goal` | (N, 2) | float32, [distancia m, ángulo rad] |
| `v`, `w` | (N,) | float32 |
| `episode_id` | (N,) | int — un id por recorrido |

**Convención del LiDAR:** `scan[i]` = rayo a `i·4°` antihorario desde el frente (`scan[0]` delante, `scan[22]` izquierda, `scan[45]` detrás). Ángulo del objetivo positivo = a la izquierda. Podéis validar vuestros ficheros con:

```python
from il_policy.contract import ContractConfig, validate_dataset
validate_dataset(dict(np.load('ruta.npz')), ContractConfig())
```
