import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from datas import ImageDataset
from torch.utils.data import DataLoader
from model import SimpleConvNet

#DO NOT FORGET!!! ADD RANDOM SEED

def get_dataset():
    mnist = MNIST(root='./dataset/data', 
                  train=True, 
                  transform=transforms.ToTensor(), 
                  download=True)
    
    #Filter out nine digits
    nine_digits = [img for img, label in mnist if label==9]

    #Use these digits to create custom dataset
    custom_dataset = ImageDataset(nine_digits)

    #Dataloader
    training_data = DataLoader(custom_dataset, batch_size=20, shuffle=True)

    return training_data


def train_model():
    data = get_dataset()

    #Fit this data into model
    model = SimpleConvNet()
    predicted_grid_cells = model(data)

    #Compare using loss function

    #Backward and update gradients

    #Optimization using adams

