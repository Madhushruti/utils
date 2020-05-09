import torch
from torchvision import transforms, datasets
import numpy as np

class DataSetInfo(object):

    def __init__(self, dataset_type, path="./data"):
        self.path = path
        self.dataset_type = dataset_type
        self.mean, self.std = self.get_mean_std()

    def get_mean_std(self):
        # simple transform
        if self.dataset_type == "mnist":
            simple_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
            exp = datasets.MNIST(self.path, train=True, download=True, transform=simple_transforms)
            exp_data = exp.train_data
            exp_data = exp.transform(exp_data.numpy())
            self.mean = torch.mean(exp_data)
            self.std =  torch.std(exp_data)
            print('[Train]')
            print(' - Numpy Shape:', exp.train_data.cpu().numpy().shape)
            print(' - Tensor Shape:', exp.train_data.size())
            print(' - min:', torch.min(exp_data))
            print(' - max:', torch.max(exp_data))
            print(' - mean:', self.mean)
            print(' - std:', self.std)
            print(' - var:', torch.var(exp_data))
        elif self.dataset_type == "cifa":
            # Note: Pending implementation
            self.mean = 0.5
            self.std = 0.5
        return self.mean, self.std

    def get_train_dataset(self, train_transform):
        if self.dataset_type == "mnist":
            train = datasets.MNIST(self.path, train=True, download=True, transform=train_transform)
        elif self.dataset_type == "cifa":
            train = datasets.CIFAR10(root=self.path, train=True,
                                                    download=True, transform=train_transform)
        return train

    def get_test_dataset(self, test_transform):
        if self.dataset_type == "mnist":
            test = datasets.MNIST(self.path, train=False, download=True, transform=test_transform)
        elif self.dataset_type == "cifa":
            test = datasets.CIFAR10(root=self.path, train=False,
                                                    download=True, transform=test_transform)
        return test

