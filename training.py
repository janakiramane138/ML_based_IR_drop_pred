
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob

class IRDropDataset(Dataset):
    def __init__(self, directory):
        self.samples = []
        for file in glob.glob(os.path.join(directory, "current_map_*.csv")):
            base = os.path.basename(file).replace("current_map_", "").replace(".csv", "")
            current = np.loadtxt(os.path.join(directory, f"current_map_{base}.csv"), delimiter=",")
            density = np.loadtxt(os.path.join(directory, f"pdn_density_map_{base}.csv"), delimiter=",")
            vsource = np.loadtxt(os.path.join(directory, f"voltage_source_map_{base}.csv"), delimiter=",")
            label = np.loadtxt(os.path.join(directory, f"ir_drop_map_{base}.csv"), delimiter=",")
            features = np.stack([current, density, vsource], axis=0)
            self.samples.append((features, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

#Unet model class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

def train_model(input_dir, model_path, epochs=30, batch_size=4, lr=0.001):
    dataset = IRDropDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, help="Directory with training CSV files")
    parser.add_argument('-output', required=True, help="Path to save trained model")
    args = parser.parse_args()

    train_model(args.input, args.output)
