import numpy as np    # Standard matrix operations
import random         # Randomizing functions
import sys, os
import time
import tensorflow as tf
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

sys.path.insert(0, "../")
sys.path.insert(0, "../../")
#from Utils.pb_utils import *
#from Utils.pb_robot_class import *
from lidar_autoencoder import *

seed = np.random.randint(0,999999)
batch_size = 20
epochs = 3
N_avg_loss_checks = 100
loss_threshold = 0.1
n_scans = 201 
both = True

parser = argparse.ArgumentParser()
parser.add_argument("-l","--latent_dim", help="Latent dimension", default=90)
args = parser.parse_args()
latent_dim = int(args.latent_dim)

sim = "DIRECT"

### INITIALIZE AUTOENCODER AND SPAWN WORLD

np.random.seed(seed)
torch.manual_seed(seed)

if both == False:
  LAE = LidarAutoencoder(n_scans=n_scans,
                         latent_dim=latent_dim,
                         capacity = 2)
else:
  LAE = LidarAutoencoder(n_scans=n_scans*2,
                         latent_dim=latent_dim,
                         capacity = 2)
LAE.to(device)  # Set to CUDA GPU or CPU
LAE.train()      # Set to training mode

optimizer = torch.optim.Adam(
  params=LAE.parameters(),
  lr = 9e-4,
  weight_decay = 9e-6)

scan_dataset = np.load("./lidar_data_both.npy")
scan_dataset = torch.Tensor(scan_dataset).unsqueeze(1)

data_loader = torch.utils.data.DataLoader(scan_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

### TRAIN MODEL
early_fin=False
train_loss_avg = []
for epoch in range(epochs):
  for ep, data in enumerate(data_loader):
    scans_batch = data.to(device)
    scans_rec,_ = LAE(scans_batch)
    # Compute error
    loss = F.mse_loss(scans_rec, scans_batch)
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    train_loss_avg.append(loss.item())
    if ep%10 == 0:
      print ("Epoch {}-{}. Loss = {:3f}".format(epoch+1,ep+1, train_loss_avg[-1]))
    if ep>N_avg_loss_checks:
      last_N = [train_loss_avg[i] for i in range(-N_avg_loss_checks,0)]
      if sum(last_N)/N_avg_loss_checks < loss_threshold:
        print ("Early finnish?")
        early_fin =True
        break
  if early_fin:
    break

### SAVE MODEL
save_path = "./cp_LAE_{:}_{:}.pth".format(n_scans, latent_dim)
if (os.path.exists(save_path)):
  if os.path.isfile(save_path):
    os.remove(save_path)
torch.save(LAE.state_dict(), save_path)

save_path = "./loss_LAE_{:}_{:}.npy".format(n_scans,latent_dim)
if (os.path.exists(save_path)):
  if os.path.isfile(save_path):
    os.remove(save_path)
np.save(save_path, np.array(train_loss_avg))