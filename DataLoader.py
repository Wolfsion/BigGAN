from enum import Enum, unique
from os.path import join

import torch
import torchvision
from torchvision import transforms
from torchgeo.datasets import UCMerced

dataset_base = "~/la/datasets"


@unique
class VDataSet(Enum):
    Init = 0
    CIFAR10 = 1
    CIFAR100 = 2
    UCM = 3
    FMNIST = 4


# CIFAR10 const config
CIFAR10_NAME = "CIFAR10"
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_NUM_TEST_DATA = 10000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# CIFAR100 const config
CIFAR100_CLASSES = 100
CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# UCM const config
UCM_CLASSES = 21
UCM_MEAN = [0.485, 0.456, 0.406]
UCM_STD = [0.229, 0.224, 0.225]


# compatible for torchgeo
def _dic2img(dic, num_classes: int):
    img = transforms.ToPILImage()(dic["image"])
    label = dic["label"]
    img = transforms.ToTensor()(img),
    img = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                               std=(0.5, 0.5, 0.5))(img)
    return img, label


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=8,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.current_iter = self.__iter__()

    def get_next_batch(self):
        try:
            return self.current_iter.__next__()
        except StopIteration:
            self.current_iter = self.__iter__()
            return self.current_iter.__next__()

    def skip_epoch(self):
        self.current_iter = self.__iter__()

    @property
    def len_data(self):
        return len(self.dataset)


def get_data(dataset: VDataSet, data_type, transform=None, target_transform=None):
    if dataset == VDataSet.CIFAR100:
        assert data_type in ["train", "test"]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        return torchvision.datasets.CIFAR100(root=join(dataset_base, "CIFAR100"),
                                             train=data_type == "train", download=True,
                                             transform=transform)
    elif dataset == VDataSet.UCM:
        assert data_type in ["train", "test", "val"]
        data_transform = transforms.Compose([
            transforms.Lambda(lambda dic: _dic2img(dic, 21)),
        ])
        ucm = UCMerced(root=join("/home/xd/la/datasets", "UCM"),
                       split=data_type, download=True,
                       transforms=data_transform,
                       checksum=False)
        return ucm
    elif dataset == VDataSet.FMNIST:
        assert data_type in ["train", "test"]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])
        return torchvision.datasets.FashionMNIST(root=join(dataset_base, "FMNIST"),
                                                 train=data_type == "train",
                                                 transform=transform, download=True)
    else:
        raise ValueError("{} dataset is not supported.".format(dataset))


def get_data_loader(name: VDataSet, data_type: str, batch_size=None, shuffle: bool = False,
                    sampler=None, transform=None, target_transform=None, subset_indices=None,
                    num_workers=8, pin_memory=False):
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        assert batch_size is not None, "Batch size for training data is required"
    if shuffle is True:
        assert sampler is None, "Cannot shuffle when using sampler"

    data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform)
    if subset_indices is not None:
        data = torch.utils.data.Subset(data, subset_indices)
    if data_type != "train" and batch_size is None:
        batch_size = len(data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=True)


if __name__ == '__main__':
    torchvision.datasets.FashionMNIST(root="~/la/datasets/FMNIST", train=True,
                                      download=True)

    torchvision.datasets.CIFAR100(root="~/la/datasets/CIFAR100", train=True,
                                  download=True)
