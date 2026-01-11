from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import torchvision.transforms.functional as TF


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path, max_count=30000):
        self.texts = []
        count = 1
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)
                    count += 1
                    if count > max_count:
                        break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ShardingLMDBDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        prompt_embeds_dir: str = "",
        image_embeds_dir: str = "",
        embeds_device: torch.device | None = None,
        embeds_dtype: torch.dtype = torch.float32,
    ):
        self.envs = []
        self.index = []
        if embeds_device is None:
            embeds_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeds_device = embeds_device
        self.embeds_dtype = embeds_dtype
        self.prompt_embeds_cache = self._load_embeddings_from_dir(
            prompt_embeds_dir, device=self.embeds_device, dtype=self.embeds_dtype
        )
        self.image_embeds_cache = self._load_embeddings_from_dir(
            image_embeds_dir, device=self.embeds_device, dtype=self.embeds_dtype
        )

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
        """
        shard_id, local_idx = self.index[idx]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        img = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "img", np.uint8, local_idx,
            shape=(832, 480, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        prompt_embeds = None
        if self.prompt_embeds_cache is not None:
            if idx not in self.prompt_embeds_cache:
                raise KeyError(f"prompt_embeds not found for idx {idx}")
            prompt_embeds = self.prompt_embeds_cache[idx]
        image_embeds = None
        if self.image_embeds_cache is not None:
            if idx not in self.image_embeds_cache:
                raise KeyError(f"image_embeds not found for idx {idx}")
            image_embeds = self.image_embeds_cache[idx]

        sample = {
            "prompts": prompts,
            "img": img,
            "idx": idx
        }
        if prompt_embeds is not None:
            sample["prompt_embeds"] = prompt_embeds
        if image_embeds is not None:
            sample["image_embeds"] = image_embeds
        return sample

    @staticmethod
    def _load_embeddings_from_dir(
        path: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if not path:
            return None
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Embeddings directory not found: {path}")
        cache = {}
        for fname in os.listdir(path):
            if not fname.endswith(".pt"):
                continue
            stem = os.path.splitext(fname)[0]
            if not stem.isdigit():
                continue
            tensor = torch.load(os.path.join(path, fname), map_location="cpu")
            cache[int(stem)] = tensor.to(device=device, dtype=dtype)
        return cache


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
