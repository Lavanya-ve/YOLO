from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

mnist = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

nine_digits = [img for img, label in mnist if label==9]

plt.imshow(nine_digits[0].squeeze(), cmap="gray")
plt.title("Image:{0}")
plt.axis("off")
plt.show()

print("Downlaoded the MNIST dataset!")
