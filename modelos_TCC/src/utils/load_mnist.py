# load_mnist.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64):
    # Transformações aplicadas ao dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Carregar dataset MNIST
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader