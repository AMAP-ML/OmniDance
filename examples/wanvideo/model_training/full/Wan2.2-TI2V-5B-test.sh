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

# CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_test.yaml examples/wanvideo/model_training/train.py \
#   --dataset_base_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data1 \
#   --dataset_metadata_path ./metadata_s2v_train2.csv \
#   --data_file_keys "video,input_audio,s2v_pose_video,audio_embeds" \
#   --num_frames 77 \
#   --height 1280 \
#   --width 704 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
#   --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train1/Wan2.2-TI2V-5B_full" \
#   --trainable_models "dit" \
#   --extra_inputs "input_image,input_audio,s2v_pose_video,audio_embeds" \
#   --save_steps 2


CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_test.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data_0206/Data \
  --dataset_metadata_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data_0206/Data/metadata_s2v_train.csv \
  --data_file_keys "video,input_audio,s2v_pose_video,audio_embeds" \
  --num_frames 77 \
  --height 1280 \
  --width 704 \
  --model_paths '[["/mnt/workspace/yangkaixing/ICML/base-v4-pmti2v/models/train/Wan2.2-TI2V-5B_full/step-48000.safetensors"], ["/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"], ["/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]]' \
  --tokenizer_path '/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl' \
  --learning_rate 5e-6 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/workspace/yangkaixing/ICML/OSS/weight/mti2v-stage2" \
  --trainable_models "dit" \
  --extra_inputs "input_image,input_audio,s2v_pose_video,audio_embeds" \
  --save_steps 2

#  --audio_processor_config '["/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/"]' \

# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_test.yaml examples/wanvideo/model_training/train.py \
#   --dataset_base_path /mnt/workspace/yangkaixing/Reference/QWenVL/Data1 \
#   --dataset_metadata_path ./metadata_s2v_train2.csv \
#   --data_file_keys "video,input_audio,s2v_pose_video,audio_embeds" \
#   --num_frames 77 \
#   --height 1280 \
#   --width 704 \
#   --model_paths '[["./models/train/Wan2.2-TI2V-5B_full/step-48000.safetensors"], ["/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"], ["/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]]' \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
#   --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train1/Wan2.2-TI2V-5B_full" \
#   --trainable_models "dit" \
#   --extra_inputs "input_image,input_audio,s2v_pose_video,audio_embeds" \
#   --save_steps 2

  # --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \

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
