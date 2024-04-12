import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class Colorizer(nn.Module):
    def __init__(self, num_downsampling_layers=5):
        super(Colorizer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_downsampling_layers - 1):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_downsampling_layers - 1):
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ))
        self.decoder = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder(x)
        
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data_path", transform=transform)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

model = Colorizer()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")

def colorize_image(input_l, output_ab):
    # Convert L*a*b* to RGB
    input_l = input_l.squeeze().numpy() * 100.0  # Scale L* channel back to [0, 100]
    output_ab = output_ab.squeeze().permute(1, 2, 0).numpy() * 127.0  # Scale a* and b* channels to [0, 255]
    lab_image = np.zeros((input_l.shape[0], input_l.shape[1], 3))
    lab_image[:, :, 0] = input_l
    lab_image[:, :, 1:] = output_ab
    rgb_image = lab2rgb(lab_image)
    return rgb_image

input_l = torch.randn(1, 1, 128, 128)
output_ab = model(input_l)
colorized_image = colorize_image(input_l, output_ab)
