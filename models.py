import torch.nn as nn

class SuperResolution(nn.Module):
    """
    Network Architecture as per specified in the paper. 
    The chosen configuration for successive filter sizes are 9-5-5
    The chosed configuration for successive filter depth are 128-64(-3)
    """
    def __init__(self, sub_image: int = 33, spatial: list = [9, 5, 5], filter: list = [128, 64], num_channels: int = 3):
        super().__init__()
        self.layer_1 = nn.Conv2d(num_channels, filter[0], spatial[0], padding = spatial[0] // 2)
        self.layer_2 = nn.Conv2d(filter[0], filter[1], spatial[1], padding = spatial[1] // 2)
        self.layer_3 = nn.Conv2d(filter[1], num_channels, spatial[2], padding = spatial[2] // 2)
        self.relu = nn.ReLU()

    def forward(self, image_batch):
        x = self.layer_1(image_batch)
        x = self.relu(x)
        x = self.layer_2(x)
        y = self.relu(x)
        x = self.layer_3(y)
        return x, y 