from __future__ import annotations

from itertools import cycle
from torch.optim import Optimizer
from torch.utils.data import (DataLoader, Dataset)
#from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import torch.onnx as tonnx

import numpy as np



#data_bird = np.load('full_numpy_bitmap_bird.npy')
#data_other = np.load('merged_array.npy')


#print(data_bird.shape)



# convert the dataset into torch tensors
#data_bird = torch.from_numpy(data_bird).float()
#print(data_bird.shape)
# reshape the dataset
#data_bird = data_bird.reshape(data_bird.shape[0], 1, 28, 28)
# add labels to the dataset
#labels_bird = torch.full((data_bird.shape[0],), 1)


#print(data_bird.shape)
#print(data_bird[0].shape)
#print(data_bird[0])
# Charger les données d'oiseaux
data_bird = np.load('full_numpy_bitmap_bird.npy')
data_not_bird = np.load('merged_array.npy')

# Convertir les données en torch.Tensor
data_bird = torch.from_numpy(data_bird).float()
data_not_bird = torch.from_numpy(data_not_bird).float()

# Redimensionner les données pour correspondre à la forme attendue (batch_size, 1, 28, 28)
data_bird = data_bird.view(-1, 1, 28, 28)
data_not_bird = data_not_bird.view(-1, 1, 28, 28)

# melanger les données
#data_bird = data_bird[torch.randperm(data_bird.size()[0])]
data_not_bird = data_not_bird[torch.randperm(data_not_bird.size()[0])]

# reduire les données pour ne pas avoir de problème de mémoire
data_bird = data_bird[:30000]
data_not_bird = data_not_bird[:30000]

# Créer les étiquettes
labels_bird = torch.full((data_bird.shape[0],), 1)
labels_not_bird = torch.full((data_not_bird.shape[0],), 0)

class Bird_or_not(Dataset):
    def __init__(self, data_bird, labels_bird, data_not_bird, labels_not_bird):
        self.data_bird = data_bird
        self.labels_bird = labels_bird
        self.data_not_bird = data_not_bird
        self.labels_not_bird = labels_not_bird
        self.bird_or_not = torch.cat((self.data_bird, self.data_not_bird), 0)
        self.labels_bird_or_not = torch.cat((self.labels_bird, self.labels_not_bird), 0)
    def __len__(self):
            return len(self.bird_or_not)
        
    def __getitem__(self, idx):
            return self.bird_or_not[idx], self.labels_bird_or_not[idx]
          



class BIRDDataset(Dataset):
        def __init__(self, train: bool, path: str, device: torch.device) -> None:
            super().__init__()
            self.path = path
            self.prefix = 'train' if train else 'test'
            self.path_xs = os.path.join(self.path, f'bird_{self.prefix}_xs.pt')
            self.path_ys = os.path.join(self.path, f'bird_{self.prefix}_ys.pt')
            self.transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])
            
            if not os.path.exists(self.path_xs) or not os.path.exists(self.path_ys):
                set = Bird_or_not(data_bird, labels_bird, data_not_bird, labels_not_bird)
                loader = DataLoader(set, batch_size=batch_size, shuffle=train)
                n = len(set)
                
                xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
                ys = torch.empty((n, ), dtype=torch.long)
                desc = f'Preparing {self.prefix.capitalize()} Set'
                for i, (x, y) in enumerate(tqdm(loader, desc=desc)):
                    xs[i * batch_size:min((i + 1) * batch_size, n)] = x
                    ys[i * batch_size:min((i + 1) * batch_size, n)] = y
                    
                torch.save(xs, self.path_xs)
                torch.save(ys, self.path_ys)

            self.device = device
            self.xs = torch.load(self.path_xs, map_location=self.device)
            self.ys = torch.load(self.path_ys, map_location=self.device)

        
        def __len__(self) -> int:
         return len(self.xs)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
         return self.xs[idx], self.ys[idx]

class FeaturesDector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = FeaturesDector(1, 32)
        self.fc = MLP(800, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler, epochs: int) -> None:
        self.train()
        batches = iter(cycle(loader))
        for _ in tqdm(range(epochs * len(loader)), desc='Training'):
            x, l = next(batches)
            optimizer.zero_grad(set_to_none=True)
            logits = self(x)
            loss = F.nll_loss(torch.log_softmax(logits, dim=1), l)
            loss.backward()
            optimizer.step()
            scheduler.step() 

    @torch.inference_mode()
    def test(self, loader: DataLoader) -> None:
        self.eval()
        loss, acc = 0, 0.0
        for x, l in tqdm(loader, total=len(loader), desc='Testing'):
            logits = self(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            loss += F.nll_loss(torch.log_softmax(logits, dim=1), l, reduction='sum').item()
            acc += (preds == l.view_as(preds)).sum().item()
        print()
        print(f'Loss: {loss / len(loader.dataset):.2e}')
        print(f'Accuracy: {acc / len(loader.dataset) * 100:.2f}%')
        print()



if __name__ == "__main__":
   from torch.optim import AdamW
   from torch.optim.lr_scheduler import OneCycleLR


   device = torch.device('cpu')
   epochs = 1
   batch_size = 512
   lr = 1e-2

   train_set = BIRDDataset(train=True, path='/home/tom/DVIC/A5/Cours/IA-CG/perso', device=device)
   train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
   
   test_set = BIRDDataset(train=False, path='/home/tom/DVIC/A5/Cours/IA-CG/perso', device=device)
   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
   
   model = CNN().to(device)
   optimizer = AdamW(model.parameters(), lr=lr, betas=(0.7, 0.9)) 
   scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=int(((len(train_set) - 1) // batch_size + 1) * epochs))
   model.fit(train_loader, optimizer, scheduler, epochs)
   model.test(test_loader)
   torch.save(model.state_dict(), 'bird_cnn.pt')

   tonnx.export(
        model.cpu(),
        torch.empty((1, 1, 28, 28), dtype=torch.float32),
        'bird.onnx',
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )



    