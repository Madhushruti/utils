import torch
from torchvision import transforms, datasets


class Transform(object):

    def __init__(self, mean=None, std=None, train_transforms = None,  test_transforms=None):
        if train_transforms is None:
            train_transforms = self.get_default_train_transforms(mean, std)
        if test_transforms is None:
            test_transforms = self.get_default_test_transforms(mean, std)
        self.test_transforms = train_transforms
        self.train_transforms = test_transforms

    def get_default_train_transforms(self, mean, std):
        train_transforms = transforms.Compose([
            #  transforms.Resize((28, 28)),
            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))  # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
            # Note the difference between (0.1307) and (0.1307,)
        ])
        return train_transforms

    def get_default_test_transforms(self, mean, std):
        # Test Phase transformations
        test_transforms = transforms.Compose([
            #  transforms.Resize((28, 28)),
            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        return test_transforms

    def get_train_transforms_rotaion(self, mean, std):
        # Train Phase transformations with rotation
        train_transforms = transforms.Compose([
                                       transforms.RandomRotation((-9.0, 9.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((mean,), (std,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       ])
        return train_transforms





