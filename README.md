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
