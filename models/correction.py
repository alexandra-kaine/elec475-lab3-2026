"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()

        # ======================================================================
        # TODO [POST-LAB STEP 2]: Implement the CorrectionNet Layers
        #
        # Requirements:
        #   1. Four conv blocks, with output channels 32, 64, 128 and 128.
        #      Each block: nn.Conv2d(kernel_size=3, padding=1) -> nn.BatchNorm2d
        #      -> ReLU -> nn.MaxPool2d(kernel_size=2, stride=2)
        #   2. fc1: in_features = 128 * 4 * 4 (2,048), out_features = 256
        #   3. fc2: in_features = 256, out_features = 2
        # ======================================================================
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 2]: Implement the Forward Pass
        #
        # 1. Apply the four conv blocks
        # 2. Flatten using torch.flatten(x, start_dim=1)
        # 3. Apply fc1 -> ReLU, then fc2 (no activation)
        # 4. Return the (B, 2) normalized offsets
        # ======================================================================
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CorrectionNetModified(nn.Module):
    """
    Your modified CorrectionNet (Post-Lab Step 5). Same input (B, 3, 64, 64) and
    output (B, 2) normalized offsets as CorrectionNet.
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement Your Modified Correction Model
        # ======================================================================
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement the Forward Pass
        # ======================================================================
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")
