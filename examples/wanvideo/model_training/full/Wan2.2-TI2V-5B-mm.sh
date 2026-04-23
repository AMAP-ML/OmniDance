# accelerate launch examples/wanvideo/model_training/train.py \
#   --dataset_base_path data/example_video_dataset \
#   --dataset_metadata_path data/example_video_dataset/metadata.csv \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Wan2.2-TI2V-5B_full" \
#   --trainable_models "dit" \
#   --extra_inputs "input_image"

export WANDB_API_KEY=b73dc7e03900ed34c234cc96dd70b0a4d59453fd
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES:$NNODES"
echo "MASTER_ADDR:$MASTER_ADDR"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_mm.yaml \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --machine_rank=$NODE_RANK --num_processes=$NUM_PROCESSES --num_machines=$NNODES \
  ./examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data1 \
  --dataset_metadata_path ./metadata_s2v_train2.csv \
  --data_file_keys "video,input_audio,s2v_pose_video,audio_embeds" \
  --num_frames 77 \
  --height 1280 \
  --width 704 \
  --model_paths '["./models/train/Wan2.2-TI2V-5B_full/step-48000.safetensors"]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train1/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,input_audio,s2v_pose_video,audio_embeds" \
  --save_steps 500

# accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
#   --dataset_base_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data1 \
#   --dataset_metadata_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data1/metadata_icml.csv \
#   --num_frames 77 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Wan2.2-TI2V-5B_full" \
#   --trainable_models "dit" \
#   --extra_inputs "input_image"
