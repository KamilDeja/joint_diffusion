"""
Train a diffusion model on images.
"""

import argparse
import copy
from collections import OrderedDict

from dataloaders import base
from dataloaders.datasetGen import *
from evaluations.validation import Validator

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from torch.utils import data
import torch.distributed as dist
from guided_diffusion.train_util import TrainLoop
from guided_diffusion import logger
import os

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
else:
    from guided_diffusion import dist_util

import torch


# os.environ["WANDB_MODE"] = "disabled"

def yielder(loader):
    while True:
        yield from loader


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)

    if args.seed is not None:
        print("Using manual seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    logger.configure()

    if logger.get_rank_without_mpi_import() == 0:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project="ddgm_representations", name=args.experiment_name, config=args)

    if args.img_size:
        args.img_size = int(args.img_size)
        train_dataset, test_set, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                          train_aug=args.train_aug,
                                                                                          resolution=args.img_size)
    else:
        train_dataset, test_set, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                          train_aug=args.train_aug)

    print(f"Training with image size:{image_size}")

    args.image_size = image_size
    args.in_channels = image_channels
    if args.dataset.lower() in ["celeba"]:
        n_classes = 40
    else:
        n_classes = train_dataset.number_classes

    args.num_classes = n_classes
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, args)

    logger.log("creating data loader...")

    if args.eval_on_test_set:
        val_dataset = test_set
    else:
        val_dataset = None

    train_dataset_split, val_dataset_split, task_output_space = data_split(dataset=train_dataset,
                                                                           val_dataset=val_dataset,
                                                                           return_classes=args.class_cond,
                                                                           return_task_as_class=False,
                                                                           limit_data=args.limit_data,
                                                                           val_size=0 if args.validate_on_train else 0.1,
                                                                           labelled_data_share=args.labelled_data_share)

    if not args.skip_validation:
        if not args.validate_on_train:
            val_data = val_dataset_split
        else:
            val_data = train_dataset_split
        val_loader = data.DataLoader(dataset=val_data,
                                     batch_size=args.microbatch if args.microbatch > 0 else args.batch_size,
                                     drop_last=True)
        # val_loaders.append(val_loader)

        stats_file_name = f"seed_{args.seed}_tasks_{args.num_tasks}_random_{args.random_split}_dirichlet_{args.dirichlet}_limit_{args.limit_data}"
        if args.use_gpu_for_validation:
            device_for_validation = dist_util.dev()
        else:
            device_for_validation = torch.device("cpu")

        validator = Validator(n_classes=n_classes, device=dist_util.dev(), dataset=args.dataset,
                              stats_file_name=stats_file_name,
                              score_model_device=device_for_validation, dataloader=val_loader,
                              multi_label_classifier=args.multi_label_classifier)
    else:
        validator = None

    train_dataset_loader = data.DataLoader(dataset=train_dataset_split,
                                           batch_size=args.batch_size, shuffle=True,
                                           drop_last=True)
    dataset_yielder = yielder(train_dataset_loader)

    if args.class_cond:
        max_class = n_classes
        # raise NotImplementedError()  # Classes seen so far for plotting and sampling
    else:
        max_class = None
    train_loop = TrainLoop(
        params=args,
        model=model,
        diffusion=diffusion,
        data=dataset_yielder,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        scheduler_rate=args.scheduler_rate,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        skip_save=args.skip_save,
        save_interval=args.save_interval,
        plot_interval=args.plot_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        num_steps=args.num_steps,
        image_size=args.image_size,
        in_channels=args.in_channels,
        class_cond=args.class_cond,
        max_class=max_class,
        validator=validator,
        validation_interval=args.validation_interval,
        semi_supervised_training=(args.labelled_data_share < 1)
    )
    if args.load_model_path is not None:
        print(f"Loading model {args.load_model_path}")
        model.load_state_dict(
            dist_util.load_state_dict(args.load_model_path + ".pt", map_location="cpu")
        )
        model.to(dist_util.dev())
        # train_loop.step = int(args.load_model_path[-8:-2])
    train_loop.run_loop()
    train_loop.plot(step=0)

    fid_result, precision, recall = validator.calculate_results(train_loop=train_loop,
                                                                dataset=args.dataset,
                                                                n_generated_examples=args.n_examples_validation,
                                                                batch_size=args.microbatch if args.microbatch > 0 else args.batch_size)

    print(f"FID: {fid_result}")


def create_argparser():
    defaults = dict(
        seed=None,
        wandb_api_key="",
        experiment_name="test",
        dataroot="data/",
        dataset="MNIST",
        schedule_sampler="uniform",
        alpha=4,
        beta=1.2,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        snr_log_interval=-1,
        num_points_plot=2,
        skip_save=False,
        save_interval=20000,
        plot_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_id=-1,
        reverse=False,
        dirichlet=None,
        num_tasks=1,
        limit_classes=-1,
        random_split=False,
        train_aug=False,
        limit_data=None,
        num_steps=10000,
        scheduler_rate=1.0,
        use_task_index=True,
        skip_validation=False,
        n_examples_validation=5000,
        validation_interval=25000,
        validate_on_train=False,
        use_gpu_for_validation=True,
        n_generated_examples_per_task=1000,
        skip_gradient_thr=-1,
        generate_previous_examples_at_start_of_new_task=False,
        generate_previous_samples_continuously=True,
        load_model_path=None,
        img_size=None,
        labelled_data_share=1.0,
        eval_on_test_set=False,
        classifier_loss_scaling=1.0,
        classifier_augmentation=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
