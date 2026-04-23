#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/mnt/workspace/yangkaixing/ICML/OSS/code/base-mti2v-test
SCRIPT=./examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py
OUT=./output/output_s3500
CKPT=/mnt/workspace/yangkaixing/ICML/OSS/weight/mti2v-stage2/step-3500.safetensors
CSV=/mnt/workspace/yangkaixing/Reference/QWenVL/Data_0206/Data/metadata_s2v_test.csv
BASE=/mnt/workspace/yangkaixing/Reference/QWenVL/Data_0206/Data
inference_mode="MTI2V"
WORLD_SIZE=8

mkdir -p "$OUT"

for i in $(seq 0 $((WORLD_SIZE-1))); do
  echo "Launching GPU $i (shard $i/$WORLD_SIZE)"
  CUDA_VISIBLE_DEVICES=$i \
  python -u "$SCRIPT" \
    --i "$i" \
    --world_size "$WORLD_SIZE" \
    --output_path "$OUT" \
    --model_checkpoint "$CKPT" \
    --csv_path "$CSV" \
    --base_dir "$BASE" \
    --mode "$inference_mode" \
    > "$OUT/log_gpu${i}.txt" 2>&1 &
done

wait
echo "All jobs finished."
