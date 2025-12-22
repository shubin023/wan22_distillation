from utils.wan_wrapper import WanVAEWrapper
import imageio.v3 as iio
from tqdm import tqdm
import argparse
import torch
import json
import math
import os
import glob

torch.set_grad_enabled(False)


def video_to_numpy(video_path):
    """
    Reads a video file and returns a NumPy array containing all frames.

    :param video_path: Path to the video file.
    :return: NumPy array of shape (num_frames, height, width, channels)
    """
    return iio.imread(video_path, plugin="pyav")  # Reads the entire video as a NumPy array


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str,
                        help="Path to the folder containing input videos.")
    parser.add_argument("--output_latent_folder", type=str,
                        help="Path to the folder where output latents will be saved.")
    parser.add_argument("--model_name", type=str, default="Wan2.1-T2V-14B",
                        help="Name of the model to use.")
    parser.add_argument("--prompt_folder", type=str,
                        help="Path to the folder containing prompt text files.")

    args = parser.parse_args()

    # Setup environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main_process = True

    # Get all video files from input folder
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.input_video_folder, ext)))
    
    # Create prompt:video pairs
    prompt_video_pairs = []
    for video_file in video_files:
        video_name = os.path.basename(video_file)
        # Replace video extension with .txt to get prompt file name
        prompt_filename = os.path.splitext(video_name)[0] + '.txt'
        prompt_file_path = os.path.join(args.prompt_folder, prompt_filename)
        
        # Check if prompt file exists
        if os.path.exists(prompt_file_path):
            # Read prompt from text file
            try:
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                # Store relative path for video
                video_relative_path = os.path.relpath(video_file, args.input_video_folder)
                prompt_video_pairs.append((prompt, video_relative_path))
            except Exception as e:
                print(f"Failed to read prompt file: {prompt_file_path}, Error: {str(e)}")
        else:
            print(f"Prompt file not found: {prompt_file_path}")

    model = WanVAEWrapper(model_name=args.model_name).to(device=device, dtype=torch.bfloat16)
    os.makedirs(args.output_latent_folder, exist_ok=True)

    # Dictionary to store video_path:latent_file_path mapping
    video_latent_map = {}
    
    # Initialize counters
    total_videos = 0
    skipped_videos = 0
    successful_encodings = 0
    failed_encodings = 0

    if is_main_process:
        print(f"processing {len(prompt_video_pairs)} prompt video pairs ...")

    # Process each prompt:video pair
    for global_index, (prompt, video_path) in enumerate(prompt_video_pairs):
        output_path = os.path.join(args.output_latent_folder, f"{global_index:08d}.pt")
        
        # Check if video file exists
        full_path = os.path.join(args.input_video_folder, video_path)
        if not os.path.exists(full_path):
            skipped_videos += 1
            continue

        # Check if we've already processed this video
        if video_path in video_latent_map:
            # If video was processed before, copy the latent to new file
            existing_dict = torch.load(video_latent_map[video_path])
            # Get the latent from the dictionary (it's the only value)
            existing_latent = next(iter(existing_dict.values()))
            torch.save({prompt: existing_latent}, output_path)
            continue

        total_videos += 1
        try:
            # Read and process video
            array = video_to_numpy(full_path)
        except Exception as e:
            print(f"Failed to read video: {video_path}")
            print(f"Error details: {str(e)}")
            failed_encodings += 1
            continue

        # Convert video to tensor and normalize
        video_tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0).permute(
            0, 4, 1, 2, 3
        ) / 255.0
        video_tensor = video_tensor * 2 - 1
        video_tensor = video_tensor.to(torch.bfloat16)

        # Encode video to latent
        encoded_latents = encode(model, video_tensor).transpose(2, 1)
        latent = encoded_latents.cpu().detach()

        # Save prompt:latent mapping
        torch.save({prompt: latent}, output_path)
        
        # Update video:latent_file mapping
        video_latent_map[video_path] = output_path
        successful_encodings += 1

        if global_index % 200 == 0:
            print(f"process {global_index} finished.")

    print("\nProcessing Statistics:")
    print(f"Total videos processed: {total_videos}")
    print(f"Skipped videos (not found): {skipped_videos}")
    print(f"Successfully encoded: {successful_encodings}")
    print(f"Failed to encode: {failed_encodings}")


if __name__ == "__main__":
    main()
