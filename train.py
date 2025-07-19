import time
import copy
import torch
import numpy as np
import pandas as pd
from barbar import Bar
from pathlib import Path
from model import ConvAutoencoder_v2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Dataset ###
class CBIRDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        image = self.transformations(Image.open(row['image']).convert('RGB'))
        return image

    def __len__(self):
        return len(self.dataFrame.index)

def prepare_data(df, random_state=0):
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=random_state)
    return CBIRDataset(train_df), CBIRDataset(val_df), CBIRDataset(test_df)

### Training ###
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    epoches_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            for inputs in Bar(dataloaders[phase]):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoches_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer_state_dict': optimizer.state_dict()
                }, f'ckpt_epoch_{epoch+1}.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)

    # Lưu biểu đồ loss
    save_loss_plot(epoches_loss, num_epochs)

    return model

### Vẽ biểu đồ loss ###
def save_loss_plot(epoches_loss, num_epochs):
    train_lss = epoches_loss[::2]
    val_lss = epoches_loss[1::2]
    itr = list(range(1, len(train_lss) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(itr, train_lss, label="Training Loss")
    plt.plot(itr, val_lss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    print("📊 Biểu đồ loss đã lưu thành 'loss_plot.png'")

### Main ###
if __name__ == '__main__':
    datasetPath = Path('dataset/')
    df = pd.DataFrame()
    df['image'] = [str(datasetPath / f) for f in os.listdir(datasetPath) if os.path.isfile(datasetPath / f)]

    train_set, val_set, test_set = prepare_data(df)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1),
        'val': DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1)
    }

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}

    model = ConvAutoencoder_v2().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=20)

    torch.save({
        'epoch': 20,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'conv_autoencoderv2_200ep.pt')

    print("✅ Training finished. Model saved as 'conv_autoencoderv2_200ep.pt'")

