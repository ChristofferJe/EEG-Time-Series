import torch
import matplotlib.pyplot as plt

def pred_training_loop(pred_model, n_epochs, loss_fn, optimizer, 
                       train_loader, validation_loader, device, scheduler, 
                       initial_teacher_forcing=1, 
                       epochs_with_teacher_forcing=20,
                       validate_every=1, 
                       plot_every=5, 
                       neptune_run=None):
    '''Takes a model, data and specified hypterparameters and trains the model and 
    validate the model over the specified amount of epochs. Returns the validation 
    loss of the last epock
    Args:
        pred_model: the model that is to be trained
        n_epochs: the number of epochs the model is going to be trained over
        loss_fn: the loss function used during training
        optimizer: the optimizer used during training
        train_loader: a torch DataLoader containing the data used during training
        validation_loader: a torch DataLoader containing the data used during validation
        device: name of the gpu to be used
        scheduler: learning rate scheduler to be used during training
        initial_teacher_forcing: the probability of teacher forcing in the intial epoch of trainig
        epochs_with_teacher_forcing: the amount of epochs to use teacher forcing in
        validate_every: how many epochs that should also validate the model
        plot_every: how many epochs should plot the train development of the model
        neptune_run: the Neptune run that should log the training, if any 
        '''
    
    def get_teacher_forcing(i_epoch, initial_teacher_forcing = initial_teacher_forcing):
        '''Calculates current teacher forcing probability for epoch i'''
        return max(0, initial_teacher_forcing * ((epochs_with_teacher_forcing - 1) - \
                                          (i_epoch - 1)) / (epochs_with_teacher_forcing - 1))

    def train(teacher_force_prob):
        '''Trains the model for one epoch using the given teacher force probability.
        Returns the average loss for the epoch.'''
        pred_model.train()
        total_train_loss = 0
        for xbatch, ybatch in train_loader:
            xbatch[0], xbatch[1], ybatch[0], ybatch[1] = (
                xbatch[0].to(device),
                xbatch[1].to(device),
                ybatch[0].to(device),
                ybatch[1].to(device)
            )
            output = pred_model(xbatch, ybatch, teacher_force_prob) 
            loss = loss_fn(output, ybatch[0])
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return total_train_loss /len(train_loader)
    
    
    def showPlot(train_loss_list, val_loss_list):
        '''Plots the train and validation loss for each concluded epoch'''
        plt.plot(train_loss_list, label = "Train loss")
        plt.plot(val_loss_list, label = "Validation loss")
        plt.legend()
        plt.show()
        
        
    def pred_validate(pred_model, loss_fn, validation_loader, device):
        '''Validates the performance of the model on the validation data.
        Returns the average loss of the model on the validation data'''
        pred_model.eval()
        total_loss = 0
        with torch.no_grad():
            for xbatch,ybatch in validation_loader:
                xbatch[0], xbatch[1], ybatch[0], ybatch[1] = (
                    xbatch[0].to(device),
                    xbatch[1].to(device),
                    ybatch[0].to(device),
                    ybatch[1].to(device)
                )
                output = pred_model(xbatch, ybatch, 0)
                total_loss += loss_fn(output, ybatch[0]).item()

        return total_loss / len(validation_loader)
    
    #Lists for printing
    train_loss_list = []
    val_loss_list = []

    for i_epoch in range(1,n_epochs+1):
        '''Train the model for the specifed number of epochs. Log the results for
        each epoch in the specified Neptune run.'''
        teacher_force_prob = get_teacher_forcing(i_epoch)
        train_loss = train(i_epoch, teacher_force_prob)
        valid_loss = pred_validate(pred_model, loss_fn, validation_loader, device)
        scheduler.step(valid_loss)

        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        neptune_run["train/loss"].append(train_loss)
        neptune_run["train/teacher_force_prob"].append(teacher_force_prob)
        neptune_run["validation/loss"].append(valid_loss)

        print(f"Epoch {i_epoch}: Training loss: {train_loss}")
        if i_epoch % validate_every == 0:
            print(f"Epoch {i_epoch}: Validation loss: {valid_loss}")
        
        if i_epoch % plot_every == 0:
            showPlot(train_loss_list, val_loss_list)
    
    return valid_loss