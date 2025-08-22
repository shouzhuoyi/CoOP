import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import os
dir = os.path.dirname(os.path.abspath(__file__))
dir_config = os.path.join(dir, '..', 'configs/transformer.yaml')

with open(dir_config, "r",encoding='utf-8') as file:
    cfg = yaml.safe_load(file)

print(cfg)
import torchvision.transforms as transforms

transform = transforms.ToTensor()
cifar10_train = torchvision.datasets.CIFAR10(
    root=cfg["dataset_root"], train=True, download=False, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    root=cfg["dataset_root"], train=False, download=False, transform=transform)


train_loader = DataLoader(
    cifar10_train, batch_size=64, shuffle=True, num_workers=2)

test_loader = DataLoader(
    cifar10_test, batch_size=64, shuffle=False, num_workers=2)

