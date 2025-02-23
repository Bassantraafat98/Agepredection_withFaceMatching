import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm

import os
import matplotlib.pyplot as plt

from Age_config import config
from Age_Detector_model import AgePredection
from Age_functions import train_one_epoch, validation
from Dataset_Loader import train_loader, valid_loader


# create a folder to save checkpoints
path = 'checkpoints'
try:
    os.mkdir(path)
except OSError as error:
    print(error)
print(torch.cuda.is_available())

# Define model, define optimizer and Set learning rate and weight decay.

model = AgePredection(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])

#For traning model VIT uncomment the next line, make the pretran_weights=False
# model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights=False).to(config['device'])

loss_fn = nn.L1Loss()
metric = tm.MeanAbsoluteError().to(config['device'])
optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])
best_loss = torch.inf
before_model_path = None

loss_train_hist = []
loss_valid_hist = []

metric_train_hist = []
metric_valid_hist = []

writer = SummaryWriter()


for epoch in range(config['epochs']):
    # Train
    model, loss_train, metric_train = train_one_epoch(model,
                                           train_loader,
                                           loss_fn,
                                           optimizer,
                                           metric,
                                           epoch)
    # Validation
    loss_valid, metric_valid = validation(model,
                                          valid_loader,
                                          loss_fn, 
                                          metric)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)

    if loss_valid < best_loss:
        best_loss = loss_valid
        if before_model_path is not None:
            os.remove(before_model_path)
        before_model_path = f'checkpoints/epoch-{epoch}-loss_valid-{best_loss:.3}.pt'
        torch.save(model.state_dict(), before_model_path)
        print(f'\nModel saved in epoch: {epoch}')

    writer.add_scalar('Loss/train', loss_train, epoch)
    writer.add_scalar('Loss/test', loss_valid, epoch)
    
    if epoch % 5 == 0:
        print()
        print(f'Train: loss={loss_train:.3}')
        print(f'Valid: loss={loss_valid:.3}')
        print()


writer.close()

# Plots: Plot learning curves
# plt.plot(range(config['epochs']), loss_train_hist, 'r-', label='Train-loss')
# plt.plot(range(config['epochs']), loss_valid_hist, 'b-', label='Validation-loss')
# plt.plot(range(config['epochs']), metric_train_hist, 'g-', label='Train-metric')
# plt.plot(range(config['epochs']), metric_valid_hist, 'b-', label='Validation-metric')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.grid(True)
# plt.legend()
# plt.savefig('./figs_Age/metric_plot.png')
