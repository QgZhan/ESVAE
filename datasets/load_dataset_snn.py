import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch
import global_v as glv


def load_mnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def load_fashionmnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_cifar10(data_path, batch_size=None, input_size=None, small=False):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_celebA(data_path, batch_size=None, input_size=None, small=False):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    test_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        transforms.CenterCrop(148),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path,
                                           split='train',
                                           download=True,
                                           transform=transform)
    testset = torchvision.datasets.CelebA(root=data_path,
                                            split='test',
                                            download=True,
                                            transform=test_transform)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size*2,
                                            shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader



