"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataloaders import base
from guided_diffusion import logger
from guided_diffusion.gaussian_diffusion import _extract_into_tensor

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
else:
    from guided_diffusion import dist_util

from guided_diffusion.script_util import (
    # NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def main():
    args = create_argparser().parse_args()
    args.num_classes = args.n_classes
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    args.model_path = f"results/{args.experiment_name}/" + args.model_path

    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    indices = list(range(diffusion.num_timesteps))[::-1]
    device = dist_util.dev()
    train_dataset, test_set, image_size, image_channels = base.CelebA("data/", train_aug=False,
                                                                      image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for _, target_class in train_loader:
        if len(all_images) * args.batch_size >= args.num_samples:
            break
        model_kwargs = {}
        target_class = target_class.float().to(dist_util.dev())
        model_kwargs["y"] = target_class
        preds_in_time = []
        img = th.randn([args.batch_size, args.in_channels, args.image_size, args.image_size], device=device)
        for i in indices:
            if i % 100 == 1:
                print(i)
            t = th.tensor([i] * img.shape[0], device=device)
            t = diffusion._scale_timesteps(t)
            h, hs = model.partial_forward(img, t)
            h.retain_grad()
            for _hs in hs:
                _hs.retain_grad()
            preds = model.clasifier(h, hs, t)

            preds_in_time.append((preds.argmax(1).detach().cpu()))
            loss = th.nn.BCEWithLogitsLoss(reduction="sum")(preds, target_class)
            loss.backward()
            model.clasifier.zero_grad()
            h = h + args.sampling_lr * h.grad
            hs_new = []
            for _hs in hs:
                hs_new.append(_hs + args.sampling_lr * _hs.grad)
            hs = hs_new
            with th.no_grad():
                model_output = model.finalise_forward(h, hs, t)
                if args.learn_sigma:
                    B, C = img.shape[:2]
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    min_log = _extract_into_tensor(
                        diffusion.posterior_log_variance_clipped, t, img.shape)
                    max_log = _extract_into_tensor(np.log(diffusion.betas), t, img.shape)
                    # The model_var_values is [-1, 1] for [min_var, max_var].
                    frac = (model_var_values + 1) / 2
                    model_log_variance = frac * max_log + (1 - frac) * min_log
                else:
                    model_log_variance = np.log(np.append(diffusion.posterior_variance[1], diffusion.betas[1:]))
                    model_log_variance = _extract_into_tensor(model_log_variance, t, img.shape)
                pred_xstart = diffusion._predict_xstart_from_eps(x_t=img, t=t, eps=model_output)
                model_mean, _, _ = diffusion.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=img, t=t
                )
                img = model_mean
                model.zero_grad()
                if i != 0:
                    img = model_mean + th.exp(0.5 * model_log_variance) * th.randn(*img.shape, device=device)
        # print(f"Sampled {args.batch_size}")
        sample = img
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [
            th.zeros_like(target_class) for _ in range(dist.get_world_size())
        ]
        # dist.all_gather(gathered_labels, gathered_labels)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(),
                                    f"samples_{args.model_path.split('/')[-1][:-3]}_with_classifier_lr_{args.sampling_lr}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    plt.figure()
    plt.axis('off')
    samples_grid = make_grid(th.from_numpy((arr.swapaxes(1, 3))), 4).permute(2, 1, 0)
    plt.imshow(samples_grid)
    out_plot = os.path.join(logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}")
    plt.savefig(out_plot)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        experiment_name="test",
        clip_denoised=False,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        gpu_id=-1,
        n_classes=10,
        train_with_classifier=True,
        sampling_lr=1000,
        learn_sigma=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
