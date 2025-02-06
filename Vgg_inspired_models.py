import torch
import torch.nn as nn
import torch.optim as optim

class CNN_V1(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        
        # First conv uses a 7x7 kernel to quickly capture wide spatial context.
        # Then standard 3x3 blocks with pooling.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Another pair of 3×3 blocks
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Another pooling
        self.pool2 = nn.MaxPool2d(2, 2)

        # Replace fully connected layer with GAP
        # GAP reduces each 32x32 feature map to 1x1, resulting in 128 outputs.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 39)  # Match the number of classes

    def forward(self, x):
        # Large kernel conv + stride
        x = nn.functional.relu(self.conv1(x))   # [batch, 32, 128, 128]
        
        # 2 small conv layers + pool
        x = nn.functional.relu(self.conv2(x))   # [batch, 64, 128, 128]
        x = nn.functional.relu(self.conv3(x))   # [batch, 64, 128, 128]
        x = self.pool(x)                        # -> [batch, 64, 64, 64]
        
        # Another 2 small conv + pool
        x = nn.functional.relu(self.conv4(x))   # [batch, 128, 64, 64]
        x = nn.functional.relu(self.conv5(x))   # [batch, 128, 64, 64]
        x = self.pool2(x)                       # -> [batch, 128, 32, 32]
        
        # Apply Global Average Pooling
        x = self.global_avg_pool(x)             # -> [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)               # Flatten -> [batch, 128]
        
        # Final linear layer for classification
        x = self.fc(x)                          # -> [batch, 39]
        return x



class CNN_V2(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN_v1, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256 * 16 * 16, 1024),  # dimension depends on input size
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 39)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class CNN_V3(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        self.conv_layers = nn.Sequential( # Sequential enables stacking multiple layers
            # Convolutional Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [batch, 32, 256, 256] #default for padding value is 0 and for the stride parameter value is 1
            nn.ReLU(),                        # padding of 1 conserves the spatial dimension
            nn.Conv2d(32, 32, kernel_size=3),           # [batch, 32, 254, 254]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # [batch, 32, 127, 127]

            # Convolutional Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# [batch, 64, 127, 127]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),           # [batch, 64, 125, 125]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # [batch, 64, 62, 62]

            # Convolutional Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [batch, 128, 62, 62]
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),          # [batch, 128, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [batch, 128, 30, 30]

            # Convolutional Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# [batch, 256, 30, 30]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),           # [batch, 256, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [batch, 256, 14, 14]

            # Convolutional Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# [batch, 512, 14, 14]
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),           # [batch, 512, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # [batch, 512, 6, 6]
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512 * 6 * 6, 1500),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1500, 39)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x