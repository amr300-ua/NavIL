#!/usr/bin/env bash
# Ciclo 1 — Módulo 3: genera datos ficticios, entrena la MLP, la evalúa y
# hace una demo de inferencia. Ejecutar desde la raíz del repositorio:
#
#   ./scripts/train_policy_dummy.sh            # con el paquete instalado (colcon)
#   USE_SOURCE=1 ./scripts/train_policy_dummy.sh   # sin compilar, desde src/
set -euo pipefail

DATA=${DATA:-data/dummy.npz}
OUT=${OUT:-models/mlp_dummy}
EPISODES=${EPISODES:-200}

if [[ "${USE_SOURCE:-0}" == "1" ]]; then
  export PYTHONPATH="$(pwd)/src/il_policy:${PYTHONPATH:-}"
  run() { python3 -m "il_policy.$1" "${@:2}"; }
  MAKE=synthetic; TRAIN=train; EVAL=evaluate; DEMO=policy
else
  run() { ros2 run il_policy "$@"; }
  MAKE=make_dummy_dataset; TRAIN=train; EVAL=evaluate; DEMO=predict_demo
fi

echo "== 1/4 Dataset ficticio =="
run "$MAKE" --out "$DATA" --episodes "$EPISODES"

echo "== 2/4 Entrenamiento =="
run "$TRAIN" --data "$DATA" --out "$OUT"

echo "== 3/4 Evaluación offline (test) =="
run "$EVAL" --model "$OUT/best.pt" --data "$DATA" --out "$OUT/eval_test.json"

echo "== 4/4 Demo de inferencia =="
run "$DEMO" --model "$OUT/best.pt"
