import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os

RECON_DIR = "reconstructions"
LOSSES_PATH = "reconstructions/losses.npy"
NUM_EPOCHS_PATH = "reconstructions/num_epochs.txt"
N_EPOCHS = 120

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64*64*3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 3, 64, 64)
        return out

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [B, 16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [B, 3, 64, 64]
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def train_autoencoder(image_path):
    os.makedirs(RECON_DIR, exist_ok=True)
    img = Image.open(image_path).resize((64,64)).convert('RGB')
    t = T.ToTensor()(img).unsqueeze(0)
    model = ConvAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        out = model(t)
        loss = loss_fn(out, t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        img_out = out.detach().squeeze().permute(1,2,0).numpy()
        img_out = (img_out * 255).astype(np.uint8)
        Image.fromarray(img_out).save(f"{RECON_DIR}/recon_{epoch}.png")
        with open(NUM_EPOCHS_PATH, "w") as f:
            f.write(str(epoch+1))
        np.save(LOSSES_PATH, np.array(losses))

def get_num_epochs():
    if os.path.exists(NUM_EPOCHS_PATH):
        with open(NUM_EPOCHS_PATH, "r") as f:
            return int(f.read())
    return 0

def get_reconstruction_path(epoch):
    return f"{RECON_DIR}/recon_{epoch}.png"

def get_losses():
    if os.path.exists(LOSSES_PATH):
        return np.load(LOSSES_PATH).tolist()
    return []
