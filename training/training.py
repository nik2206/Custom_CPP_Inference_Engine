# -*- coding: utf-8 -*-
"""training.ipynb"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------------
# 1. ARCHITECTURE CONTRACT COMPLIANCE
# -------------------------------------------------------------
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Input: 1 x 28 x 28
        # Conv2D: 8 filters, 3x3 kernel, stride 1, padding 0 -> Output: 8 x 26 x 26
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)

        # MaxPool: 2x2, stride 2 -> Output: 8 x 13 x 13
        # Flattened size = 8 * 13 * 13 = 1352
        self.fc = nn.Linear(in_features=8 * 13 * 13, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten to 1D array
        x = self.fc(x)
        return x

def main():
    # Save directly to Colab's default working directory
    models_dir = "/content"

    print("Step 1: Loading Fashion-MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStep 2: Training the FP32 baseline model for 5 epochs...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/5 - Loss: {running_loss / len(trainloader):.4f}")

    print("\nTraining complete!")

    # -------------------------------------------------------------
    # 3. EXPORT MODEL CONTRACT
    # -------------------------------------------------------------
    print("\nStep 3: Exporting weights and biases to JSON...")
    model.eval()
    state_dict = model.state_dict()

    # PRECISION FIX: Removed .numpy() to maintain raw tensor precision
    model_json = {
        "conv_weight": state_dict["conv.weight"].cpu().flatten().tolist(),
        "conv_bias": state_dict["conv.bias"].cpu().flatten().tolist(),
        "fc_weight": state_dict["fc.weight"].cpu().flatten().tolist(),
        "fc_bias": state_dict["fc.bias"].cpu().flatten().tolist()
    }

    model_path = os.path.join(models_dir, "fp32_model.json")
    with open(model_path, "w") as f:
        json.dump(model_json, f)
    print(f"Saved: {model_path}")

    # -------------------------------------------------------------
    # 4. EXPORT TEST IMAGE CONTRACT
    # -------------------------------------------------------------
    print("\nStep 4: Exporting a single test sample for C++ verification...")
    image, label = next(iter(testloader))

    with torch.no_grad():
        image_device = image.to(device)
        raw_output = model(image_device)
        
        # PRECISION FIX: Removed .numpy()
        logits = raw_output.cpu().flatten().tolist()

    test_image_json = {
        "label": int(label.item()),
        "data": image.flatten().tolist(),
        "expected_logits": logits
    }

    image_path = os.path.join(models_dir, "test_image.json")
    with open(image_path, "w") as f:
        json.dump(test_image_json, f)
    print(f"Saved: {image_path}")
    print("\nDay 1 Python tasks complete! The files are ready to download.")

if __name__ == "__main__":
    main()