import torch
import torch.nn as nn
from torch.nn import Module
# from datas import ImageDataset
from torchvision.datasets import MNIST
from torchvision import transforms

class SimpleConvNet(Module):

    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pred_head = nn.Conv2d(in_channels=64,out_channels=5, kernel_size=1, stride=1)

    def forward(self, x):
        y1 = self.conv_model(x)
        y2 = self.pred_head(y1)
        return y2
    
# if __name__ == "__main__":

#     mnist = MNIST(
#             root="./data",
#             train=True,
#             download=True,
#             transform=transforms.ToTensor()
#         )

#     nine_digits = [img for img, label in mnist if label==9]

#     custom_dataset = ImageDataset(nine_digits)
#     img, target = custom_dataset[12]
#     torch.reshape(img,(64,64,1)) 
#     # print("The shape of the image is: ", img.shape)

#     simpleConv = SimpleConvNet()
#     res = simpleConv(img)
#     print(res.shape)