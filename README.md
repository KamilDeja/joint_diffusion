# Joint Diffusion

This is the codebase for [Learning Data Representations with Joint Diffusion Models](https://arxiv.org/abs/2301.13622).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for joint modeling with classifier
# Training models

Training diffusion models is based on the original training from [openai repository](https://github.com/openai/improved-diffusion) with few modifications

```
mpiexec -n N python scripts/classifier_train.py --dataroot data/ --experiment_name experiment_name $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

All generated artifacts will be stored in `results/$experiment_name` Make sure to include external data (such as ImageNet) in `dataroot`, for standard benchmarks in torchvision it will be downloaded automatically
Make sure to divide the batch size in `TRAIN_FLAGS` by the number of MPI processes you are using.

Additional flags for joint training:
- `--train_with_classifier` - Use classifier for training True/False
- `--train_noised_classifier ` - Whether to train classifier on noised samples or only the original data samples
- `--eval_on_test_set ` - Whether to evaluate on the final test set
- `--classifier_loss_scaling ` - Scaling the classifier loss - how much loss from a classifier we want to add to the final loss
- `--multi_label_classifier` - Whether to train a multilabel classifier instead of multiclass

For sampling and evaluation, follow instructions in [openai repository](https://github.com/openai/improved-diffusion)

For sampling with classifier there are two additional scripts
`image_sample_with_classifier.py` and `image_sample_with_classifier_CelebA.py`
They use additional parameters:
- `sampling_lr` - parameter alpha used for conditional generation


## Reproducibility
Training commands to reproduce the main results from the publication:

SVHN:
```
--dataset SVHN --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule linear --use_kl False --lr 1e-4 --batch_size 256 --schedule_sampler uniform --plot_interval 10000 --save_interval 20000 --class_cond True --skip_save False --gpu_id 1 --validation_interval 100 --num_steps 250000 --skip_validation False --validate_on_train False --train_with_classifier True --train_noised_classifier False --eval_on_test_set True --classifier_loss_scaling 0.001
```

CIFAR10:
```
--dataset CIFAR10 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule linear --use_kl False --lr 1e-4 --batch_size 256 --schedule_sampler uniform --plot_interval 10000 --save_interval 20000 --class_cond True --skip_save False --gpu_id 1 --validation_interval 100 --num_steps 500000 --skip_validation False --validate_on_train False --train_with_classifier True --train_noised_classifier False --eval_on_test_set True --classifier_loss_scaling 0.001
```

CIFAR100:
```
--dataset CIFAR100 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule linear --use_kl False --lr 1e-4 --batch_size 256 --schedule_sampler uniform --plot_interval 10000 --save_interval 20000 --class_cond True --skip_save False --gpu_id 1 --validation_interval 100 --num_steps 500000 --skip_validation False --validate_on_train False --train_with_classifier True --train_noised_classifier False --eval_on_test_set True --classifier_loss_scaling 0.001
```

CelebA:
```
--dataset FashionMNIST --num_channels 64 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 500 --noise_schedule linear --use_kl False --lr 1e-4 --batch_size 128 --schedule_sampler uniform --plot_interval 10000 --save_interval 20000 --class_cond True --skip_save False --gpu_id 1 --validation_interval 100 --num_steps 100000 --skip_validation False --validate_on_train False --train_with_classifier True --train_noised_classifier False --eval_on_test_set True --classifier_loss_scaling 0.001
```

For smaller GPUs, use the --microbatch to accumulate gradients after several passes of microbatch
