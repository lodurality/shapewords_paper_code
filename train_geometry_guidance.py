#rougly based off textual inversion huggingface example https://huggingface.co/docs/diffusers/en/training/text_inversion

import argparse
import logging
import math
import os
import time

import diffusers
import numpy as np
import open_clip
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        # why this
        "linear": PIL.Image.BILINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

from dataset import ULIPControlNetDataset
from geometry_guidance_models import Shape2CLIP

# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )

    parser.add_argument(
        "--use_after_obj_only",
        action="store_true",
        help="Use eos and obj token",
        default=False,
    )

    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        required=False,
        help="Path to training split, if any",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--textured_images_dir", type=str,
        default=None,
    )
    parser.add_argument(
        "--prompts_path", type=str,
        default=None,
    )
    parser.add_argument(
        "--depth_images_dir", type=str,
        default="/scratch/workspace/dmpetrov_umass_edu-ulip/shapenet-55/only_rgb_depth_images",
    )
    parser.add_argument(
        "--pb_emb_path", type=str,
        default="/scratch/workspace/dmpetrov_umass_edu-ulip/shapenet_pointbert",
    )

    parser.add_argument(
        "--categories_path", type=str,
        default="/home/dmpetrov/work_data/token_geometry/shapenet_pointbert_1000/categories.json",
    )
    parser.add_argument('--use_pb_embs',action='store_true')
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument("--num_depth_views", type=int, default=4, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="projection_model_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    ## gg model params
    parser.add_argument(
        "--gg_drop_path_rate",
        type=float,
        default=0,
        help="Dropout for attn",
    )
    parser.add_argument(
        "--gg_depth",
        type=int,
        default=4,
        help="GG CA depth",
    )
    parser.add_argument(
        "--gg_heads",
        type=int,
        default=8,
        help="GG CA heads",
    )

    parser.add_argument(
        "--train_prompt_padding",
        type=str,
        default='max_length',
        help="how to pad train prompts",
    )

    parser.add_argument(
        "--min_t_share",
        type=float,
        default=0.0,
        help="Min percentage steps",
    )

    parser.add_argument(
        "--max_t_share",
        type=float,
        default=1.0,
        help="Max percentage steps",
    )

    parser.add_argument(
        "--loss_weight_m",
        type=float,
        default=500.0,
        help="M parameter for weighting function",
    )

    parser.add_argument(
        "--loss_weight_s",
        type=float,
        default=125.0,
        help="M parameter for weighting function",
    )

    parser.add_argument(
        "--weight_loss_by_t",
        action='store_true',
        default=False,
        help="Whether weight loss",
    )
    ##
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--from_memory", action="store_true", help="Whether to load the data from the memory")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=50,
        help=(
            "saving the projection model"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--only_debugging_shapes",
        action='store_true',
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--use_transform",
        action="store_true",
        help="Whether to use augmentation for SDS", default=False
    )
    parser.add_argument(
        "--use_clip_residual",
        action="store_true",
        help="Whether to use augmentation for SDS", default=False
    )

    #new
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=100000
    )
    parser.add_argument(
        "--max_shapes",
        type=int,
        default=10e6
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--min_crop_scale",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--flip",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_crop_scale",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--pad_border",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_pb_tokens",
        action="store_true",
        help="Whether to use PB tokens", default=False
    )

    parser.add_argument("--test",action="store_true")

    args, unknown = parser.parse_known_args()
    print(f'{unknown=}')
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args










debug_shapes = [
    '03001627_1bec15f362b641ca7350b1b2f753f3a2',
    '03001627_1157d8d6995da5c0290d57214c8512a4',
    '03001627_1f5a2c231265aa9380b3cfbeccfb24d2',
    '03001627_2a417b5a946ff7eb2a3f8f484e6c5c4f',
    '03001627_8e2b44ec14701d057c2b071b8bda1b69']

def main():
    #starting main and accelerator
    args = parse_args()
    args.output_dir = f"{args.output_dir}_{args.exp_name if args.exp_name != None else ''}_{args.train_batch_size}_{args.num_depth_views}_{args.learning_rate}{'_slr' if args.scale_lr else '' }_{args.num_train_epochs}_{args.gradient_accumulation_steps}_{args.lr_scheduler}_UT={args.use_transform}"
    if args.test:
        assert os.path.exists(args.output_dir)
    # args.textured_images_dir = os.path.join(args.textured_images_dir,debug_shapes[args.chunk_id])
    # args.output_dir = os.path.join(args.output_dir,debug_shapes[args.chunk_id])
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir) # 1. initializing the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    

    print(args)
    if args.tokenizer_name: #2. initializing the models
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)

    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    if args.weight_loss_by_t:
        test_t = np.array(list(range(noise_scheduler.config.num_train_timesteps)))
        m = args.loss_weight_m
        s = args.loss_weight_s
        vals = np.exp(-1 * (test_t - m) ** 2 / (2 * s ** 2))
        weight_alpha = noise_scheduler.alphas_cumprod[test_t]
        upweight_t = torch.sqrt((1 - weight_alpha) / weight_alpha)
        final_weights = vals * upweight_t.numpy()
        final_weights = final_weights / final_weights.max()
        final_weights = torch.FloatTensor(final_weights)
        #print(final_weights)
    else:
        final_weights = torch.ones_like(noise_scheduler.alphas_cumprod)

    #print(final_weights)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    
    pb_dim = 384 if args.use_pb_tokens else 768
    guidance_model = Shape2CLIP(pb_dim=pb_dim, heads = args.gg_heads,
                                       depth=args.gg_depth, drop_path_rate=args.gg_drop_path_rate)
    
    #3.  Freeze vae, unet, clip_model,text_encoder
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_model.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.scale_lr: #4. scale_lr
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes 
        )
    
    #5. initialize the optimizer 

    optimizer = torch.optim.AdamW(
    guidance_model.parameters(),  # only optimize the projection matrix
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,)


    # 6. Dataset
    split_ids = np.loadtxt(args.split_path, dtype=str, delimiter='\t') if args.split_path else None

    train_dataset = ULIPControlNetDataset(
        textured_images_root=args.textured_images_dir,
        preprocess=preprocess,
        size=args.resolution,
        repeats=args.repeats,
        flip_p=args.flip,
        max_images = args.max_images,
        max_shapes = args.max_shapes,
        min_crop_scale = args.min_crop_scale,
        max_crop_scale = args.max_crop_scale,
        pad_border = args.pad_border,
        use_transform = args.use_transform,
        use_pb_embs=args.use_pb_embs,
        pb_emb_path=args.pb_emb_path,
        prompts_path=args.prompts_path,
        from_memory=args.from_memory,
        categories_path=args.categories_path,
        split_ids=split_ids)

    print("DATASET")
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
    )
    # 7. initialise the scheduler
    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,)

    #8. prepare the model
    guidance_model, optimizer,train_dataloader, lr_scheduler = accelerator.prepare(
    guidance_model, optimizer,train_dataloader, lr_scheduler)

    #9. get the weight dataypes
    print(f'{accelerator.mixed_precision=}')
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    #10. move the model to device and weight datatype
    clip_model.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    final_weights = final_weights.to(accelerator.device, dtype=weight_dtype)

    # 11. get the number of epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs*num_update_steps_per_epoch

    # 12. logger and resume_checkpoint
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("depth_projection", config=vars(args))
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)} {len(train_dataloader)=}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}") # TODO
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    logger.info(f"  max_train_steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint: #TODO: check this
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,)

    # 13. original_cls_token 
    training_prompt = 'a photo of a'
    inference_prompt = 'a photo of a'
    training_prompt_tokens = torch.LongTensor(tokenizer.encode(training_prompt, padding='max_length')).to(accelerator.device)
    inference_prompt_tokens = torch.LongTensor(tokenizer.encode(inference_prompt, padding='max_length')).to(accelerator.device)
    print(f"{training_prompt_tokens.shape=}")
    #text_encoder(s)
    with torch.no_grad():
        training_prompt_emb = text_encoder(training_prompt_tokens.unsqueeze(0)).last_hidden_state.detach().clone()
        inference_prompt_emb = text_encoder(inference_prompt_tokens.unsqueeze(0)).last_hidden_state.detach().clone()
    original_cls_token = training_prompt_emb[0,0,:].unsqueeze(0).clone()
    inference_cls_token = inference_prompt_emb[0,0,:].unsqueeze(0).clone()
    print(f'{original_cls_token.shape=}')

    if args.test:
        accelerator.end_training()
        test(args,clip_model,preprocess,accelerator,original_cls_token=inference_cls_token,testing_images_dir='/scratch/workspace/dmpetrov_umass_edu-ulip/geometry-editing-2d3d/debug_images')
        return

    # 14 starting training
    start_time = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        guidance_model.train()
        print(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(guidance_model):
                batch['textured_images'] = batch['textured_images'].to(dtype=weight_dtype)
                #batch['depth_images'] = batch['depth_images'].to(dtype=weight_dtype)
                batch['pb_embs'] = batch['pb_embs'].to(dtype=weight_dtype)
                #print(batch['prompts'])
                #print(batch['pb_embs'].shape)
                #print(batch['textured_images'].shape)
                #batch_size,num_views,input_channels,depth_resolution = batch['depth_images'].shape[0],batch['depth_images'].shape[1],batch['depth_images'].shape[2],batch['depth_images'].shape[3]
                #batch['depth_images'] = batch['depth_images'].reshape(shape=(batch_size*num_views,input_channels,depth_resolution,depth_resolution))


                # Convert depth images to latent space and textured images to clip images features
                textured_features_original = clip_model.encode_image(batch['textured_images'])
                latents = vae.encode(batch['textured_images']).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                sample_t_min = int(args.min_t_share*noise_scheduler.config.num_train_timesteps)
                sample_t_max = int(args.max_t_share*noise_scheduler.config.num_train_timesteps)
                #timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = torch.randint(sample_t_min, sample_t_max, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                #depth_proj = projection_model(batch['pb_embs']).to(dtype=weight_dtype) # the graphs are broken here
                #prompt_tokens = [torch.LongTensor(tokenizer.encode(item, padding=args.train_prompt_padding)).to(
                #    accelerator.device).unsqueeze(0) for item in batch['prompts']]
                prompt_tokens = tokenizer(batch['prompts'], padding=args.train_prompt_padding).input_ids
                prompt_tokens = torch.LongTensor(prompt_tokens).to(accelerator.device)
                #print(prompt_tokens.shape)
                #prompt_tokens = torch.cat(prompt_tokens, dim=0)
                #print(prompt_tokens.shape)
                eos_inds = torch.where(prompt_tokens == 49407)[1]
                object_inds = eos_inds-1
                #print(prompt_tokens)
                #print(prompt_tokens)
                #print(prompt_tokens.shape)
                # text_encoder(s)
                with torch.no_grad():
                    prompt_emb = text_encoder(
                        prompt_tokens).last_hidden_state.detach().clone()

                #print(prompt_emb.shape, batch['pb_embs'].shape)
                #print(batch['pb_embs'].shape)
                if len(batch['pb_embs'].shape) == 2 and not args.use_pb_tokens:
                    pb_embs = batch['pb_embs'].unsqueeze(1)
                elif len(batch['pb_embs'].shape) == 3 and args.use_pb_tokens:
                    pb_embs = batch['pb_embs']
                else:
                    raise ValueError("Safety check: PB embs dimension should match use_pb_tokens")
                #print(pb_embs.shape)

                guidance_delta = guidance_model(prompt_emb, pb_embs)
                #print('HERE?')
                if args.use_after_obj_only:
                    fin_guidance = torch.zeros_like(guidance_delta)
                    for cur_ind in range(len(guidance_delta)):
                        obj_word = batch['category'][cur_ind]
                        encoded_category = tokenizer.encode(obj_word)
                        obj_word_token = encoded_category[-2]
                        obj_inds = torch.where(prompt_tokens[cur_ind] == obj_word_token)[0]
                        fin_guidance[cur_ind, eos_inds[cur_ind]:, :] = guidance_delta[cur_ind, eos_inds[cur_ind]:, :]
                        fin_guidance[cur_ind, obj_inds,:] = guidance_delta[cur_ind, obj_inds,:]
                else:
                    #print('using all_embs')
                    fin_guidance = guidance_delta
                encoder_hidden_states = prompt_emb + fin_guidance.half()

                #print(encoder_hidden_states.dtype, noisy_latents.dtype)
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                #print(model_pred.shape)

                #loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss_weights = final_weights[timesteps][:, None, None, None]#.to(accelerator_device)
                #print(loss_weights.shape, loss_weights, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss * loss_weights
                loss = loss.mean(axis=(1,2,3)).mean()
                #loss
                #print(loss.shape)

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if (global_step % (num_update_steps_per_epoch*args.save_epochs)) == 0: # changed this
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    
                    if (global_step % (num_update_steps_per_epoch*args.save_epochs)) == 0: # changed this
                        projection_model_save_path = os.path.join(args.output_dir, "projection_model-{:07d}.pth".format(global_step))
                        unwraped_guidance_model = accelerator.unwrap_model(guidance_model)
                        torch.save(unwraped_guidance_model.state_dict(),projection_model_save_path)


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # print(logs,global_step)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        projection_model_save_path = os.path.join(args.output_dir, f"projection_model.pth")
        unwrapped_guidance_model = accelerator.unwrap_model(guidance_model)
        torch.save(unwrapped_guidance_model.state_dict(),projection_model_save_path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    accelerator.end_training()
    #test(args,clip_model,preprocess,accelerator,original_cls_token=inference_cls_token,testing_images_dir='/scratch/workspace/dmpetrov_umass_edu-ulip/geometry-editing-2d3d/debug_images')
    return args
    





def test(args,clip_model,preprocess,accelerator,original_cls_token,testing_images_dir=''):
    print("in the test funtion")
    if args.use_pb_embs:
        assert args.pb_emb_path is not None, "pb_emb_path should not be None"
        path_template = '{}/{}'
        pb_dict = {}
        all_emb_files = sorted(os.listdir(args.pb_emb_path))
        for cur_file in all_emb_files:
            cur_data = np.load(path_template.format(args.pb_emb_path, cur_file))
            cur_dict = dict(zip(cur_data['ids'], cur_data['embs']))
            pb_dict = {**pb_dict, **cur_dict}
    print("loaded the pb dict")
    


    if accelerator.is_main_process:
        print(f"{accelerator.device=}")
        projection_model = ProjectionModelPBOnly(input_size=1024,channels=[1024,1024,1024,1024],act_layer=nn.GELU()) #non linear projection
        projection_model.load_state_dict(torch.load(os.path.join(args.output_dir,"projection_model-010000.pth")))
        # print(f'{projection_model.layers[0].weight[0,:25]=}')
        # print(f'{projection_model.layers[2].weight[0,:25]=}')
        # print(f'{projection_model.projection_matrix[0,:25]=}')
        projection_model.to(accelerator.device,dtype=torch.float16)
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to(accelerator.device)
        with torch.no_grad():
            for i, image in enumerate(os.listdir(testing_images_dir)):
                testing_image = os.path.join(testing_images_dir,image)
                testing_image_head = os.path.basename(testing_image[:-4])
                testing_image_id = os.path.basename(testing_image)
                testing_image = Image.open(testing_image).convert('RGB')
                testing_image_id = testing_image_id.split("_")
                shape_id = f"{testing_image_id[1]}_{testing_image_id[2]}"
                print(f'{shape_id=}')
                pb_embs = torch.from_numpy(pb_dict[shape_id]).to(accelerator.device,dtype=torch.float16).unsqueeze(0)
                testing_image_input = preprocess(testing_image.convert('RGB')).to(accelerator.device,dtype=torch.float16)
                print(f'{testing_image_input.shape=}')
                testing_features_original = clip_model.encode_image(testing_image_input.unsqueeze(0))
                print(f"{testing_features_original.shape=}{testing_features_original.dtype=}{testing_features_original.device=}{pb_embs.shape=} {pb_embs.dtype}=")

                depth_projection = projection_model(pb_embs).to(dtype=torch.float16)
                # depth_projection *= 1/testing_features_original.norm(dim=-1,keepdim=True))
                test_proj_emb = torch.cat((original_cls_token.clone(),depth_projection.expand(76,-1))).unsqueeze(0)
                print(test_proj_emb.shape)
                seed_value = 0
                generator = torch.Generator(device=accelerator.device).manual_seed(seed_value)
                with torch.no_grad():
                    learned_images = pipeline(prompt_embeds=test_proj_emb, num_inference_steps=50,num_images_per_prompt=5,generator=generator).images
                for i,learned_image in enumerate(learned_images):
                    image_path = os.path.join(args.output_dir,"validation",testing_image_head)
                    os.makedirs(image_path, exist_ok=True)
                    learned_image.save(os.path.join(image_path,f"{i}.png"))
        del pipeline
        projection_model.train



if __name__ == "__main__":
    args = main()
    
  



