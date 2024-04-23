from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle
import scipy

from models import BiasDataset, BPNet, ATACloss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train(config):

    #Load the data
    #TODO load training + validation
    dataset = BiasDataset('../results/background_GC_matched.pkl', '../results/ATAC_background.pkl')
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, num_workers=2)

    #Initialize model, loss, and optimizer
    biasModel = BPNet()
    biasModel.to(device)

    criterion = ATACloss(weight_MSE=config["weight_MSE"])
    optimizer = torch.optim.Adam(biasModel.parameters(), lr=config["lr"])
    
    train_loss, val_losses = [], []
    for epoch in range(config["nb_epoch"]):
        
        biasModel.train() 
        running_loss = 0

        for data in tqdm(dataloader):
            inputs, _, _, tracks = data 
            inputs = torch.reshape(inputs, (-1,4,2114)).to(device)
            tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

            optimizer.zero_grad()

            x, profile, count = biasModel(inputs)
            
            loss = criterion(tracks, profile, count)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            break

        epoch_loss = running_loss / len(dataloader)
        train_loss.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{config["nb_epoch"]}], Loss: {epoch_loss:.4f}')

        #TODO Compute loss on validation set here + evaluate model
        val_loss = 0.0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, _, _, tracks = data 
                inputs = torch.reshape(inputs, (-1,4,2114)).to(device)
                tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

                x, profile, count = biasModel(inputs)

                #Compute loss
                loss = criterion(tracks, profile, count)
                val_loss += loss.cpu().numpy()

                #Compute evaluation metrics: pearson correlation
                counts_per_item = torch.sum(tracks, dim=1)
                corr_tot = [scipy.stats.pearsonr(x.item(), counts_per_item[i].item())[0] for i,x in enumerate(count)] 


                The per-base read count track is evaluated using the Jensen-Shannon divergence distance,
                 which computes the divergence between two probability distributions; in this case 
                 the actual per base read profile for the 1000bp region and the predicted per base 
                 read profile for the 1000bp region.


        val_losses.append(val_loss)

    print('Finished Training')

    return biasModel, train_loss, val_losses

""" config = {
    "weight_MSE": tune.choice([2 ** i for i in range(9)]),
    "nb_epoch": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16,32,64])
} """

config = {
    "weight_MSE": 1,
    "nb_epoch": 1,
    "lr": 0.004,
    "batch_size": 32
}

biasModel, train_loss, val_losses = train(config)

##TODO setup hyperparametrs tuning ray tune
...

with open('../results/loss_train_bias.pkl', 'wb') as file:
    pickle.dump(train_loss, file)

with open('../results/loss_val_bias.pkl', 'wb') as file:
    pickle.dump(val_losses, file)