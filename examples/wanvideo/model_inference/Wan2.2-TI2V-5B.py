import argparse
import torch
from PIL import Image
from diffsynth.data import save_video, VideoData, save_video_with_audio
# from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import pandas as pd
from tqdm import tqdm
import os
import math
import pickle
import librosa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--model_checkpoint", type=str, required=True)

    # 可选：推理参数
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--world_size", type=int, default=8)
    p.add_argument("--mode", type=str, default='TI2V')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, args.mode), exist_ok=True)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path="/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=args.model_checkpoint),
            ModelConfig(path="/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path="/mnt/workspace/yangkaixing/ICML/OSS/model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    )
    df = pd.read_csv(args.csv_path)
    n = len(df)
    chunk = (n + args.world_size - 1) // args.world_size  # 向上取整
    start = args.i * chunk
    end = min(start + chunk, n)
    df = df.iloc[start:end]

    for i, row in tqdm(df.iterrows()):
        image_path = os.path.join(args.base_dir, row["input_image"])
        input_image = Image.open(image_path).convert("RGB")

        video_name = os.path.basename(image_path).replace('.png', '.mp4')
        video_path = os.path.join(args.output_path, args.mode, video_name)
        if os.path.exists(video_path):
            continue
        text = row["prompt"]
        pose_video_path = os.path.join(args.base_dir, row["s2v_pose_video"])
        s2v_pose_video = VideoData(pose_video_path, height=1280, width=704).raw_data()

        audio_embed_path = os.path.join(args.base_dir, row["audio_embeds"])
        with open(audio_embed_path, "rb") as f:
            audio_embeds = pickle.load(f)['music'][:77, :]

        audio_path = os.path.join(args.base_dir, row["input_audio"])    
        input_audio, sample_rate = librosa.load(audio_path, sr=16000)
        
        video = pipe(
            prompt=text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=0, tiled=False,
            height=1280, width=704,
            audio_sample_rate=sample_rate,
            input_image=input_image,
            s2v_pose_video=s2v_pose_video,
            input_audio=input_audio,
            audio_embeds=audio_embeds,
            num_frames=77,
            num_inference_steps=50,
            cfg_scale=5.0,
            mode=args.mode
        )
        save_video_with_audio(video, video_path, audio_path, fps=16, quality=5)


if __name__ == "__main__":
    main()