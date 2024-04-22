from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle

from models import BiasDataset, BPNet, ATACloss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, num_epochs, dataloader):
    
    losses = []
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0

        for data in tqdm(dataloader):
            inputs, time, cell_type, tracks = data 
            inputs = torch.reshape(inputs, (-1,4,2114)).to(device)
            tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

            optimizer.zero_grad()

            x, profile, count = model(inputs)
            
            loss = criterion(tracks, profile, count)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')

    return losses



dataset = BiasDataset('../results/background_GC_matched.pkl', '../results/ATAC_background.pkl')
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True, num_workers=0)

biasModel = BPNet().to(device)
criterion = ATACloss(weight_MSE=1)
optimizer = torch.optim.Adam(biasModel.parameters(), lr=1e-4)

loss = train(biasModel, criterion, optimizer, 3, dataloader)

with open('../results/loss_test.pkl', 'wb') as file:
    pickle.dump(loss, file)