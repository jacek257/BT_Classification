import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models
# import datasets
from tqdm import tqdm
# import copy
# import helpers


def evaluate_model(model, loader, criterion, device, pbar=None):
    """Evaluates the accuracy and loss of model on the loader dataset. Loss
    is evaluated using criterion

    Parameters
    ----------
    model : torch.nn
        The model to evaluate
    loader : torch.utils.data.dataloader.DataLoader
        The dataset loader to evaluate on
    criterion : torch.nn.modules.loss
        the loss function
    device : str
        The training device label. 'cuda' for gpu
    pbar : tqdm.std.tqdm
        optional progress bar object to iterate

    Returns
    -------
    tuple(float *2)
        A tuple containing the accuracy and loss/error repsectively

    """

    #set model to evaluation mode
    model.eval()

    eval_loss = 0
    eval_acc = 0

    with torch.no_grad():
        for (inputs, labels) in loader:
            labels = labels.float().unsqueeze(1)
            # send data to device
            inputs, labels = (inputs.to(device), labels.to(device))

            # forward
            outputs = model(inputs)

            # get acc and error
            _, predictions = torch.max(outputs, 1)
            eval_acc += torch.sum(predictions == labels)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            if(pbar):
                pbar.update(1)


    return (eval_acc.cpu().numpy()/len(loader.dataset),
            eval_loss/len(loader.dataset))

def train_model(model, num_epochs, train_loader, valid_loader, criterion,
        optimizer, device):
    """Trains the model and logs the evalutaions of the model every epoch
    Parameters
    ----------
    model : torch.nn
        the model to train and evaluate. The best training iteration is saved
    num_epochs : int
        the number of epochs to train
    train_loader : torch.utils.data.dataloader.DataLoader
        the DataLoader for the training set
    valid_loader : torch.utils.data.dataloader.DataLoader
        the DataLoader for the validation set
    criterion : torch.nn.modules.loss
        the loss function
    optimizer : torch.optim
        the gradient optimizer
    device : str
        the inference device label. 'cuda' for gpu

    Returns
    -------
    a 5 tuple containing the following in their respective order
        - A list of the training accuracy wrt. to epoch
        - A list of the training error wrt. to epoch
        - A list of the validation accuracy wrt. to epoch
        - A list of the validation error wrt. to epoch
        - The best performing model
    """
    LOG_ta = []
    LOG_te = []
    LOG_va = []
    LOG_ve = []
    best_model = None

    # set model to training mode
    model.train()
    pbar = tqdm(total=num_epochs*(2*len(train_loader) + len(valid_loader)))
    for epoch in range(num_epochs):
        for (inputs, labels) in train_loader:
            # send data to device
            labels = labels.float().unsqueeze(1)
            inputs, labels = (inputs.to(device), labels.to(device))

            # zero gradient, then forward
            optimizer.zero_grad()
            outputs = model(inputs)
#             print(labels, outputs)
            # backwards
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # increment progress bar
            pbar.update(1)

        # evaluate model on validation and trainig sets
        ta, te = evaluate_model(model, train_loader, criterion, device, pbar)
        va, ve = evaluate_model(model, valid_loader, criterion, device, pbar)

        # log evaluation
        LOG_ta.append(ta)
        LOG_te.append(te)
        LOG_va.append(va)
        LOG_ve.append(ve)

        if(len(LOG_ve) >= 1 or LOG_ve[-1] < LOG_ve[-2]):
            best_model = copy.deepcopy(model)

    pbar.close()
    return LOG_ta, LOG_te, LOG_va, LOG_ve, best_model
