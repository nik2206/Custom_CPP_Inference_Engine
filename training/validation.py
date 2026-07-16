# -*- coding: utf-8 -*-
"""validation.ipynb"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training for 5 epochs to hit 85% accuracy benchmark...")
model.train()
for epoch in range(5):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5 Complete.")

print("\nEvaluating on 10,000 test images...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Get the index of the highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Model Accuracy: {accuracy:.2f}%")

if accuracy >= 85.0:
    print("SUCCESS: Model meets the 85% accuracy contract!")
else:
    print("WARNING: Model is under 85%. You may need to train for a few more epochs.")

print("\nExporting the 10-image validation set for the C++ engine...")
models_dir = "/content"
validation_data = []

with torch.no_grad():
    for i in range(10):
        image, label = testset[i]
        image_device = image.unsqueeze(0).to(device)

        raw_output = model(image_device)
        
        # PRECISION FIX: Removed .numpy() here
        logits = raw_output.cpu().flatten().tolist()

        validation_data.append({
            "id": i,
            "label": int(label),
            # PRECISION FIX: Removed .numpy() here too
            "data": image.flatten().tolist(),
            "expected_logits": logits
        })

val_path = os.path.join(models_dir, "validation_set.json")
with open(val_path, "w") as f:
    json.dump(validation_data, f)

print(f"Saved 10 test images to: {val_path}")
print("Day 2 Python tasks complete! Download validation_set.json and hand it off.")