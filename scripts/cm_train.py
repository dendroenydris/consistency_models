"""
Train a diffusion model on images.
"""

from PIL import Image
from torchvision import datasets, transforms
import argparse
import os

import torch
import wandb

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
from cm.in32_data import load_dataset
import torch.distributed as dist
import copy


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.pth_out)

    save_dataset(image_size=32, data_dir=args.data_dir, image_size=args.image_size)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.to(dist_util.dev())
    model = dist_util.wrap_model(model)
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(f"warning, using smaller global_batch_size of { dist.get_world_size()*batch_size} instead of {args.global_batch_size}")
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    # data = load_dataset(args.data_dir)
    data = dist_util.get_dataloader(data)

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(
            model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(
                args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )

    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    wandb.login(key="63ce76eebffb80b1165fb79e11d6dbb677cb7db6")
    wandb.init(
        project="consistency-model-training",
        config={
            # "learning_rate": args.total_training_steps,
            "architecture": "CM",
            "dataset": "Imagenet-32",
            "epochs": args.total_training_steps,
        }
    )

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        pth_out="",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def save_images(dataset, output_dir, image_size=32):
    """
    Saves a dataset's images into a specified directory.
    Images are resized to the given size.
    """
    for idx, (image, label) in enumerate(dataset):
        # Resize the image
        image = transforms.ToPILImage()(image)
        image = image.resize((image_size, image_size),
                             Image.Resampling.LANCZOS)

        # Save the image as PNG with label and index in filename
        image_path = os.path.join(output_dir, f"label_{label}_image_{idx}.png")
        image.save(image_path, format="PNG")

    print(f"Data saved in '{output_dir}'.")


def save_dataset(image_size=32, base_dir="./dataset/MNIST"):
    """
    Saves the MNIST training and validation datasets into the specified directory.
    """
    # Define the train and validation directories
    # train_dir = os.path.join(base_dir, "train")
    # val_dir = os.path.join(base_dir, "val")

    # Create the directories if they don't exist
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)

    # Download and load the MNIST datasets
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    # val_dataset = datasets.MNIST(
    #     root="./data", train=False, download=True, transform=transform)

    # Save the datasets
    save_images(train_dataset, output_dir=base_dir, image_size=image_size)
    # save_images(val_dataset, output_dir=val_dir, image_size=image_size)


if __name__ == "__main__":
    main()
