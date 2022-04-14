from torchvision import transforms
from torchvision.datasets import mnist


class MNIST(mnist.MNIST):
    def __init__(
        self,
        rotated: bool = False,
        root: str = "./data",
        download: bool = True,
        *args,
        **kwargs
    ):
        transform = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if rotated:
            transform.append(transforms.RandomRotation(degrees=45))
        transform = transforms.Compose(transform)
        super().__init__(
            root=root, transform=transform, download=download, *args, **kwargs
        )
