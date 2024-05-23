def train_loop(data, optimizer, biasModel, criterion):

    inputs, tracks = data 
    inputs = inputs.to(device)
    tracks = tracks.to(device)

    optimizer.zero_grad()

    _, profile, count = biasModel(inputs)
            
    loss, MNLLL, MSE = criterion(tracks, profile, count)

    loss.backward() 
    optimizer.step()

    running_loss += loss.item()
    running_MNLLL += MNLLL.item()
    running_MSE += MSE.item()

    #print every 2000 batch the loss
    epoch_steps += 1
    if i % 2000 == 1999:  # print every 2000 mini-batches
    print(
          "[%d, %5d] loss: %.3f"
          % (epoch + 1, i + 1, running_loss / epoch_steps)
        )