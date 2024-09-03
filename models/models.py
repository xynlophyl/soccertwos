import torch
from torch import nn

class LinearModel(nn.Module):

    """
    Simple Linear Net
    """
    
    def __init__(
        self,
        in_channels,
        out_size
    ):
      
        super().__init__()
        
        self.layer1 = nn.Linear(in_channels, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, out_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        
        x = self.fc(x)
        
        return x
    
class CNN(nn.Module):

    """
    Implementation of a Convolutional Neural Network adapted from Deepmind's research of DeepRL in Atari
    """

    def __init__(
        self,
        in_channels,
        out_size
    ):
        
        super().__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(in_channels, 16, 8)
        self.layer2 = nn.Conv2d(16, 32, 4)
        self.layer3 = nn.Linear(32, 256)
        self.fc = nn.Linear(256, out_size)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.fc(x)

        return x


