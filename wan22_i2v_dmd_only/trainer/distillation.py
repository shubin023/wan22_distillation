import gc  # manual garbage collection to control GPU/CPU memory pressure
import logging  # lightweight logging for GC events

# Data utilities: LMDB loader for I2V, text loaders for T2V, and an infinite iterator
from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset, TextFolderDataset

# Misc helpers: deterministic seeding and merging multiple log dicts
from utils.misc import (
    set_seed,
    merge_dict_list
)

from omegaconf import OmegaConf  # config container/merging
from model import DMD  # the only distribution-matching model we keep in this project
import torch  # tensor ops and optimizers
import wandb  # optional experiment logging
import time  # simple wall-clock timing
import os  # filesystem utilities
from torch.nn.utils import clip_grad_norm_


class Trainer:
    """
    Distributed DMD trainer for Wan2.2 I2V (high- or low-noise stage).
    Handles rank setup, model/FSDP wrapping, dataloaders, EMA, checkpointing, and the
    alternating generator/critic updates used by DMD.
    """
    def __init__(self, config):
        self.config = config  # hydra/omegaconf config merged in train.py
        self.step = 0  # global training step counter

        # Step 1: Initialize device/dtype, logging/seed for single-GPU training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32  # compute dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = True  # single-process training
        self.causal = config.causal  # kept for API parity; not used in this DMD-only fork
        self.disable_wandb = config.disable_wandb  # whether to skip wandb logging

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            config.seed = random_seed.item()

        set_seed(config.seed)  # seed for reproducibility

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir  # where checkpoints are written

        # Step 2: Build model (DMD student/teacher/critic) and wrap with FSDP
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss: only 'dmd' is supported in this project")

        # Move ancillary modules to device
        if self.config.training_target == "low_noise":
            self.high_noise_step_list = torch.tensor(config.high_noise_step_list, dtype=torch.long, device=self.device)

        if self.config.i2v:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16)

        elif not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Optimizers: split params by adapter role to avoid cross-updating LoRAs
        gen_params = [p for n, p in self.model.generator.named_parameters()
                      if p.requires_grad and "lora_" in n and "_generator" in n]
        critic_params = [p for n, p in self.model.fake_score.named_parameters()
                         if p.requires_grad and "lora_" in n and "_critic" in n]

        self.generator_optimizer = torch.optim.AdamW(
            gen_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        ) if gen_params else None

        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        ) if critic_params else None

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))  # LMDB shards with prompts/latents/img
        else:
            if self.config.data_type == "text_folder":
                data_max_count = config.get("data_max_count", 30000)
                dataset = TextFolderDataset(config.data_path, data_max_count)
            elif self.config.data_type == "text_file":
                dataset = TextDataset(config.data_path)
            else:
                raise ValueError("Invalid data type")
            
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8)

        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "resume_ckpt", False):
            print(f"Resuming training from {config.resume_ckpt}")
            
            # Set resume step
            if getattr(config, "resume_step", False):
                self.step = config.resume_step
                print(f"Resuming from step {self.step}")

            # Load generator checkpoint
            # Load LoRA adapters if present
            lora_path = os.path.join(config.resume_ckpt, "model.pt")
            if os.path.exists(lora_path):
                print(f"Loading LoRA adapters from {lora_path}")
                lora_state = torch.load(lora_path, map_location="cpu")
                gen_lora = lora_state.get("generator_lora", {})
                critic_lora = lora_state.get("critic_lora", {})
                if gen_lora:
                    self.model.generator.load_state_dict(gen_lora, strict=False)
                    print("Generator LoRA loaded successfully")
                if critic_lora:
                    self.model.fake_score.load_state_dict(critic_lora, strict=False)
                    print("Critic LoRA loaded successfully")
            else:
                print(f"Warning: LoRA checkpoint not found at {lora_path}")
        

        ##############################################################################################################

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering model states...")
        # Save only the LoRA adapter weights for generator/critic roles (shared backbone stays frozen)
        gen_lora = {k: v for k, v in self.model.generator.state_dict().items() if "lora_" in k and "_generator" in k}
        critic_lora = {k: v for k, v in self.model.fake_score.state_dict().items() if "lora_" in k and "_critic" in k}
        state_dict = {
            "generator_lora": gen_lora,
            "critic_lora": critic_lora,
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        """
        One forward/backward pass for either generator or critic.
        - Builds conditioning (text, optional image latent).
        - For low-noise stage, runs the frozen high-noise model to get x_bound.
        - Computes loss/gradients for generator or critic depending on train_generator flag.
        """
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Encode text (and image for I2V) and, for low-noise stage, run the frozen high-noise model to get boundary latent
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            if self.config.i2v:
                img = batch["img"].to(self.device).squeeze(0)
                y = self.model.vae.run_vae_encoder(img)
            else:
                y = None

            if self.config.training_target == "low_noise":
                noise = torch.randn(
                    image_or_video_shape, 
                    device=self.model.device, 
                    dtype=self.model.dtype
                )
                noisy_image_or_video = noise
                for index, current_timestep in enumerate(self.high_noise_step_list):
                    timestep_id = 1000 - torch.ones(
                        noise.shape[:2],
                        device=noise.device,
                        dtype=torch.int64
                    ) * current_timestep
                    flow_pred, noisy_image_or_video = self.model.high_noise_model(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        # timestep=timestep,
                        timestep_id=timestep_id,
                        y=y
                    )
                    if index != len(self.high_noise_step_list) - 1:
                        next_timestep_id = 1000 - self.high_noise_step_list[index + 1] * torch.ones(
                            noise.shape[:2], 
                            dtype=torch.long, 
                            device=noise.device
                        )
                        noisy_image_or_video = self.model.scheduler.add_noise_high(
                            noisy_image_or_video.flatten(0, 1),
                            torch.randn_like(noisy_image_or_video.flatten(0, 1)),
                            # flow_pred.flatten(0, 1),
                            # next_timestep.flatten(0, 1),
                            next_timestep_id.flatten(0, 1),
                            self.model.high_noise_model.timestep_bound
                        ).unflatten(0, noisy_image_or_video.shape[:2])
                self.model.x_bound = noisy_image_or_video

        # Step 3: Compute and backprop generator loss if this step is a generator update
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                y=y
            )

            torch.cuda.empty_cache()

            generator_loss.backward()
            if self.generator_optimizer is not None:
                generator_grad_norm = clip_grad_norm_(
                    [p for p in self.model.generator.parameters() if p.requires_grad],
                    self.max_grad_norm_generator
                )
            else:
                generator_grad_norm = torch.tensor(0.0, device=self.device)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Otherwise compute and backprop critic loss (fake_score learns the score field)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            y=y
        )

        critic_loss.backward()
        if self.critic_optimizer is not None:
            critic_grad_norm = clip_grad_norm_(
                [p for p in self.model.fake_score.parameters() if p.requires_grad],
                self.max_grad_norm_critic
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        """Utility for quick qualitative eval: run the training sampler to produce a video."""
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        # Use the same training pipeline to denoise into latents/pixels
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        """
        Main training loop: alternates generator and critic updates,
        periodically saves checkpoints and logs to wandb.
        """
        start_step = self.step

        while True:
            if self.is_main_process:
                print(f"training step {self.step} ...")
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Alternate updates: generator every dfake_gen_update_ratio steps, critic every step
            if TRAIN_GENERATOR and self.generator_optimizer is not None:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()

            # Train the critic
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
                critic_log_dict = merge_dict_list(extras_list)
                self.critic_optimizer.step()
            else:
                critic_log_dict = {}

            # Increment the step since we finished gradient update
            self.step += 1

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                if critic_log_dict:
                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
