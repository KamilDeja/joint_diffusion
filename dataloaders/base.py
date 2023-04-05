import os

import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset
from dataloaders.image_datasets import ImageDataset, _list_image_files_recursively


class FastCelebA(Dataset):
    def __init__(self, data, attr):
        self.dataset = data
        self.attr = attr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.attr[index]


class FastDataset(Dataset):
    def __init__(self, data, labels, num_classes):
        self.dataset = data
        self.labels = labels
        self.number_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]


def CelebA(root, skip_normalization=False, train_aug=False, image_size=64, target_type='attr'):
    transform = transforms.Compose([

        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print("Loading data")
    save_path = f"{root}/fast_celeba_{image_size}"
    if os.path.exists(save_path):
        print(f"Loading from {save_path}")
        fast_celeba = torch.load(save_path)
    else:
        dataset = torchvision.datasets.CelebA(root=root, download=True, transform=transform,
                                              target_type=target_type)
        print(f"{save_path} not found, creating new")
        train_loader = DataLoader(dataset, batch_size=len(dataset))
        data = next(iter(train_loader))
        fast_celeba = FastCelebA(data[0], data[1])
        torch.save(fast_celeba, save_path)
    # train_set = CacheClassLabel(train_set)
    # val_set = CacheClassLabel(val_set)
    return fast_celeba, None, 64, 3


def SVHN(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.SVHN(
        root=dataroot,
        split="train",
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.SVHN(
        dataroot,
        split="test",
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3


def MNIST32(dataroot, skip_normalization=False, train_aug=False):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        download=True,
        transform=transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3

def USPS(dataroot, skip_normalization=False, train_aug=False):
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Grayscale(3),
        torchvision.transforms.Resize(28),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5)),
    ])

    train_dataset = torchvision.datasets.USPS(
        root=dataroot,
        train=True,
        download=True,
        transform=transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.USPS(
        dataroot,
        train=False,
        download=True,
        transform=transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 28, 1

def DA_SVHN_MNIST(dataroot, skip_normalization=False, train_aug=False):
    train_dataset_SVHN, val_dataset_SVHN, resolution, channels = SVHN(dataroot, skip_normalization, train_aug)
    train_dataset_MNIST, val_dataset_MNIST, resolution_MNIST, channels_MNIST = MNIST32(dataroot, skip_normalization,
                                                                                       train_aug)
    if (channels_MNIST != channels) or (resolution != resolution_MNIST):
        raise Exception("Wrong number of channels or wrong resolution")
    # train_dataset_MNIST.dataset.labels = np.ones_like(train_dataset_MNIST.dataset.labels) - 2
    train_dataset_MNIST.dataset.targets = np.ones_like(train_dataset_MNIST.dataset.targets) - 2
    train_dataset = ConcatDataset([train_dataset_SVHN, train_dataset_MNIST])
    # val_dataset = ConcatDataset([val_dataset_SVHN, val_dataset_MNIST])
    train_dataset.number_classes = 10
    return train_dataset, val_dataset_MNIST, resolution, channels


def MNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 28, 1


def Omniglot(dataroot, skip_normalization=False, train_aug=False):
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(1, -1)
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(1, -1)
        ])

    train_transform = val_transform

    train_dataset = torchvision.datasets.Omniglot(
        root=dataroot,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    print("Using train dataset for validation in OMNIGLOT")
    return train_dataset, train_dataset, 28, 1


def FashionMNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))  # for  28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 28, 1


def DoubleMNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for  28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset_fashion = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    # train_dataset_fashion = CacheClassLabel(train_dataset_fashion)

    train_dataset_mnist = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    # train_dataset_mnist = CacheClassLabel(train_dataset_mnist)

    val_dataset_fashion = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    # val_dataset_fashion = CacheClassLabel(val_dataset)

    val_dataset_mnist = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    # val_dataset_mnist = CacheClassLabel(val_dataset)
    train_dataset_mnist.targets = train_dataset_mnist.targets + 10
    val_dataset_mnist.targets = val_dataset_mnist.targets + 10
    train_dataset = ConcatDataset([train_dataset_fashion, train_dataset_mnist])
    train_dataset.root = train_dataset_mnist.root
    val_dataset = ConcatDataset([val_dataset_fashion, val_dataset_mnist])
    val_dataset.root = val_dataset_mnist.root
    val_dataset = CacheClassLabel(val_dataset)
    train_dataset = CacheClassLabel(train_dataset)
    return train_dataset, val_dataset, 28, 1


def CIFAR10(dataroot, skip_normalization=False, train_aug=False):
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3


def CERN(dataroot, skip_normalization=False, train_aug=True, test_split=0.25):
    data_cond = np.load(f'{dataroot}/cern/data_nonrandom_particles.npz')["arr_0"]
    data_cond = pd.DataFrame(data_cond, columns=['Energy', 'Vx', 'Vy', 'Vz', 'Px', 'Py', 'Pz', 'mass', 'charge'])
    data = np.load(f'{dataroot}/cern/data_nonrandom_responses.npz')["arr_0"]
    n_classes = 10
    bin_labels = list(range(n_classes))
    data_cond["label"] = pd.qcut(data_cond['Energy'], q=n_classes, labels=bin_labels)
    data = np.log(data + 1)
    data = np.expand_dims(data, 1)
    train_cond = data_cond.sample(int(len(data_cond) * (1 - test_split)))
    test_cond = data_cond.drop(train_cond.index)

    train_dataset = TensorDataset(torch.Tensor(data[train_cond.index]).float(),
                                  torch.Tensor(train_cond["label"].values.astype(int)).long())
    test_dataset = TensorDataset(torch.Tensor(data[test_cond.index]).float(),
                                 torch.Tensor(test_cond["label"].values.astype(int)).long())

    train_dataset.root = dataroot
    train_dataset = CacheClassLabel(train_dataset)
    test_dataset.root = dataroot
    test_dataset = CacheClassLabel(test_dataset)
    raise NotImplementedError()  # Check size
    return train_dataset, test_dataset


def Flowers(dataroot, skip_normalization=False, train_aug=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    size = 64
    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        if skip_normalization:
            train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(100),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(100),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    train_dir = dataroot + "/flower_data/train/"
    val_dir = dataroot + "/flower_data/valid/"
    # train_dir = dataroot + "/flowers_selected/"
    # val_dir = dataroot + "/flowers_selected/"
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    # If doesn't work please download data from https://www.kaggle.com/c/oxford-102-flower-pytorch

    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=train_transform)

    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 64, 3


def OFFICE(dataroot, source, skip_normalization=False, train_aug=False, image_size=64):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Resize(image_size),
        normalize,
    ])
    train_dir = f"{dataroot}/office31/{source}/"
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset)
    return train_dataset, train_dataset, image_size, 3


def OFFICE_A(dataroot, skip_normalization=False, train_aug=False):
    return OFFICE(dataroot, "amazon", skip_normalization, train_aug)

def OFFICE_W(dataroot, skip_normalization=False, train_aug=False):
    return OFFICE(dataroot, "webcam", skip_normalization, train_aug)

def OFFICE_D(dataroot, skip_normalization=False, train_aug=False):
    return OFFICE(dataroot, "dslr", skip_normalization, train_aug)


def DA_OFFICE_A_W(dataroot, skip_normalization=False, train_aug=False):
    train_dataset_A, val_dataset_A, resolution, channels = OFFICE_A(dataroot, skip_normalization, train_aug)
    train_dataset_W, val_dataset_W, _, _ = OFFICE_W(dataroot, skip_normalization, train_aug)

    # train_dataset_MNIST.dataset.labels = np.ones_like(train_dataset_MNIST.dataset.labels) - 2
    train_dataset_W.dataset.targets = np.ones_like(train_dataset_W.dataset.targets) - 2
    train_dataset = ConcatDataset([train_dataset_A, train_dataset_W])
    # val_dataset = ConcatDataset([val_dataset_SVHN, val_dataset_MNIST])
    train_dataset.number_classes = 31
    return train_dataset, val_dataset_W, resolution, channels



def CIFAR100(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3


def LSUN(dataroot, skip_normalization=False, train_aug=False):
    dataset_dir = dataroot + "lsun_bedroom/"
    all_files = _list_image_files_recursively(dataset_dir)
    resolution = 256
    train_dataset = ImageDataset(image_paths=all_files, resolution=resolution, classes=np.zeros(len(all_files)))

    return train_dataset, train_dataset, resolution, 3


def MNIST_mini(dataroot, skip_normalization=False, train_aug=False):
    dataset_dir = dataroot + "mnist_limited/"
    all_files = _list_image_files_recursively(dataset_dir)
    resolution = 28
    train_dataset = ImageDataset(image_paths=all_files, resolution=resolution, classes=np.zeros(len(all_files)))

    return train_dataset, train_dataset, resolution, 1


def ImageNet(dataroot, skip_normalization=False, train_aug=False, resolution=64):
    resolution = int(resolution)
    dataset_dir = dataroot + "ImageNet/train"
    all_files = _list_image_files_recursively(dataset_dir)
    print(f"Total number of files:{len(all_files)}")
    train_dataset = ImageDataset(image_paths=all_files, resolution=resolution, classes=np.zeros(len(all_files)))

    return train_dataset, train_dataset, resolution, 3


def FastImageNet(root, skip_normalization=False, train_aug=False, image_size=64):  # Requires a lot of memory
    transform = transforms.Compose([

        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print("Loading data")
    save_path = f"{root}/fast_imagenet_{image_size}"
    if os.path.exists(save_path):
        print(f"Loading from {save_path}")
        fast_imagenet = torch.load(save_path)
    else:
        dataset, _, _, _ = ImageNet(dataroot=root, resolution=image_size)
        print(f"{save_path} not found downloading")
        train_loader = DataLoader(dataset, batch_size=len(dataset))
        data = next(iter(train_loader))
        fast_imagenet = FastDataset(data[0], data[1], num_classes=1000)
        torch.save(fast_imagenet, save_path)
    return fast_imagenet, None, image_size, 3


def CIFAR10AUG(dataroot, skip_normalization=False, train_aug=False):
    dataset_dir = dataroot + "cifar_train/"
    all_files = _list_image_files_recursively(dataset_dir)
    resolution = 32
    train_dataset = ImageDataset(image_paths=all_files, resolution=resolution, classes=np.zeros(len(all_files)))

    return train_dataset, train_dataset, resolution, 3


def Malaria(dataroot, skip_normalization=False, train_aug=False):
    dataset_dir = dataroot + "cell_images"
    all_files = _list_image_files_recursively(dataset_dir)
    resolution = 64
    classes = np.array(["Parasitized" in file for file in all_files]).astype(int)
    train_dataset = ImageDataset(image_paths=all_files, resolution=resolution, classes=classes)

    return train_dataset, train_dataset, resolution, 3


def _Birds(dataroot, skip_normalization=False, train_aug=False, image_size=64):
    all_files = pd.read_csv(dataroot + "CUB_200_2011/images_selected.csv")
    all_files = dataroot + "CUB_200_2011/images/" + all_files["1"].values
    classes = torch.load(dataroot + "CUB_200_2011/attributes/preprocessed_attributes.th")
    train_dataset = ImageDataset(image_paths=all_files, resolution=image_size, classes=classes)
    train_dataset.number_classes = classes.size(1)
    return train_dataset, train_dataset, image_size, 3


def Birds(dataroot, skip_normalization=False, train_aug=False, image_size=64):
    print("Loading data")
    save_path = f"{dataroot}/fast_birds_{image_size}"
    if os.path.exists(save_path):
        print(f"Loading from {save_path}")
        fast_birds = torch.load(save_path)
    else:
        dataset, _, _, _ = _Birds(dataroot, skip_normalization, train_aug, image_size)
        print(f"{save_path} not found, creating new")
        train_loader = DataLoader(dataset, batch_size=len(dataset))
        data = next(iter(train_loader))
        fast_birds = FastDataset(data[0], data[1], dataset.number_classes)
        torch.save(fast_birds, save_path)
    return fast_birds, fast_birds, image_size, 3
