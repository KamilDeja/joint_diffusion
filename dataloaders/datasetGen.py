import numpy as np
import torch

from random import shuffle

from torch.utils.data import Subset

from .wrapper import Subclass, AppendName


def data_split(dataset, val_dataset=None, return_classes=False, return_task_as_class=False,
               limit_data=None, val_size=0.2, seed=0, labelled_data_share=1):
    rng = np.random.default_rng(seed=seed)
    res = []
    if val_dataset:
        val_size = 0
    num_selected = int(val_size * len(dataset))
    train_set_indices_bitmask = torch.ones(len(dataset))
    validation_indices = rng.choice(range(len(dataset)), num_selected, replace=False)
    train_set_indices_bitmask[validation_indices] = 0
    if limit_data:
        num_selected = int((1 - float(limit_data)) * len(dataset))
        skipped_indices = rng.choice(range(len(dataset)), num_selected, replace=False)
        train_set_indices_bitmask[skipped_indices] = -1

    train_subset = Subset(dataset, torch.where(train_set_indices_bitmask == 1)[0])
    if val_dataset:
        val_subset = val_dataset
    else:
        val_subset = Subset(dataset, torch.where(train_set_indices_bitmask == 0)[0])

    train_dataset_split = AppendName(train_subset, [0] * len(train_subset), return_classes=return_classes,
                                     return_task_as_class=return_task_as_class,
                                     labelled_data_share=labelled_data_share)
    val_dataset_split = AppendName(val_subset, [0] * len(train_subset), return_classes=return_classes,
                                   return_task_as_class=return_task_as_class)

    print(
        f"Prepared dataset with splits: {len(train_dataset_split)}")
    print(
        f"Validation dataset with splits: {len(val_dataset_split)}")

    return train_dataset_split, val_dataset_split, None
