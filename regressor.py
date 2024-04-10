import torch
import torch.nn as nn

class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
      
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(3, 2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x

def scale_input(input_data):
    return input_data / 100.0

model = SimpleRegressor()

input_data = torch.randn(1, 1, 128, 128)
input_data_scaled = scale_input(input_data)
print("Scaled input shape:", input_data_scaled.shape)

output = model(input_data_scaled)
print("Output shape:", output.shape)
