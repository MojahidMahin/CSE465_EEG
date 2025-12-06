import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# =========================================
# 1. INSERT ALL MODEL DEFINITIONS FROM NOTEBOOK 2
# =========================================

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, Ch=62, T=3000):
        super().__init__()
        # (Replace with your Notebook 2 EEGNet code)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (T - 63), num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, Ch=62, T=3000):
        super().__init__()
        # (Replace with DeepConvNet code)
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25 * (T - 9), num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes=2, Ch=62, T=3000):
        super().__init__()
        # (Replace with ShallowConvNet code)
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 13)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40 * (T - 12), num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================================
# 2. Build MODEL FACTORY
# =========================================

MODEL_FACTORY = {
    "EEGNet": EEGNet,
    "DeepConvNet": DeepConvNet,
    "ShallowConvNet": ShallowConvNet,
}

print("Loaded models:", list(MODEL_FACTORY.keys()))

# =========================================
# 3. LOAD SEED-VIG DATASET
# =========================================

X = np.load("X_seedvig.npy")
Y = np.load("Y_seedvig.npy")

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================================
# 4. TRAINING LOOP
# =========================================

def train_model(model, loader, epochs=3):
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# =========================================
# 5. TRAIN ALL MODELS
# =========================================

results = {}

for name, model_class in MODEL_FACTORY.items():
    print("\n=============================")
    print(f"Training {name}")
    print("=============================")

    model = model_class(num_classes=2)
    train_model(model, loader, epochs=3)
    results[name] = "Trained successfully"

print("\nFinal Results:")
print(results)
