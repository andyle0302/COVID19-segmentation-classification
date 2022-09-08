# importing the libraries
import numpy as np
from tqdm import tqdm
import torch


def training_loop(
    model, optimizer, criterion, scheduler, device, num_epochs,
        dataloader, checkpoint_path, model_path):

    model.to(device)
    # List to store loss to visualize
    lossli = []
    accli = []

    # track change in validation loss
    valid_loss_min = np.Inf
    count = 0

    # nếu val_loss tăng 8 lần thì ngừng
    patience = 8

    for epoch in range(num_epochs):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0.0
        valid_acc = 0.0

        ###################
        # train the model #
        ###################

        model.train()

        # cxr
        # for data, label in tqdm(dataloader['train']):

        # lung
        # for _, _, _, data, _, label, _ in tqdm(dataloader['train']):

        # cxrnotlung
        for _, _, _, _, data, label, _ in tqdm(dataloader['train']):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*data.size(0)

            _, pred = torch.max(output, 1)

            train_acc += pred.eq(label).sum().item()

        scheduler.step()

        ######################
        # validate the model #
        ######################

        model.eval()
        with torch.no_grad():

            # cxr
            # for data, label in tqdm(dataloader['val']):

            # lung
            # for _, _, _, data, _, label, _ in tqdm(dataloader['val']):

            # notlung
            for _, _, _, _, data, label, _ in tqdm(dataloader['val']):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item()*data.size(0)

                # Calculate accuracy
                _, pred = torch.max(output, 1)
#                 y_true += target.tolist()
#                 y_pred += pred.tolist()

                valid_acc += pred.eq(label).sum().item()

        # calculate average losses
        train_loss = train_loss / len(dataloader['train'].dataset)
        valid_loss = valid_loss / len(dataloader['val'].dataset)
        lossli.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss})

        train_acc = train_acc*100/len(dataloader['train'].dataset)
        valid_acc = valid_acc*100/len(dataloader['val'].dataset)
        accli.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'valid_acc': valid_acc})

        ####################
        # Early stopping #
        ##################

        # print training/validation statistics
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \
            \n \tTraining Acc: {:.6f} \tValidation Acc: {:.6f}'.format(
                epoch, train_loss, valid_loss, train_acc, valid_acc))
        # save model if validation loss has decreased

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': accli,
            'loss_list': lossli,
            'loss': loss
            }, checkpoint_path)

        if valid_loss <= valid_loss_min:
            print(
                'Val loss decreased ({:.6f} --> {:.6f}). Saving model'.format(
                    valid_loss_min,
                    valid_loss))

            count = 0
            print('count = ', count)
            torch.save(model, model_path)  # save model

            valid_loss_min = valid_loss
        else:
            count += 1
            print('count = ', count)
            if count >= patience:
                print('Early stopping!')

                return lossli, accli

    return lossli, accli
