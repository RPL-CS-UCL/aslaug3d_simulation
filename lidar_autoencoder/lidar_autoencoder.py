import pybullet as p  # PyBullet simulation package
import numpy as np    # Standard matrix operations
import random         # Randomizing functions
import sys
import time
import tensorflow as tf
import torch 
import torch.nn as nn
import torch.nn.functional as F
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def weights_init(m):
  if isinstance(m, nn.Conv2d):
    torch.nn.init.xavier_uniform_(m.weight)
    if m.bias:
      torch.nn.init.xavier_uniform_(m.bias)

def weights_init_2(m):
  if isinstance(m, nn.Conv2d):
    torch.nn.init.normal_(m.weight)
    if m.bias:
      torch.nn.init.normal_(m.bias)

class LidarEncoder(nn.Module):
  def __init__(self, capacity = 8, 
              n_scans = 90, latent_dim=10):
    super(LidarEncoder,self).__init__()
    c = capacity #capacity of conv1
    # IN: [Batch x 1 x 1 x N_scans]
    self.conv1 = nn.Conv1d (
      in_channels  = 1,
      out_channels = c,
      kernel_size  = 11)
    # C1: [B x c x (N-(11//2)*2)]
    # C1: [B x c x (N-10)]
    self.conv2 = nn.Conv1d (
      in_channels  = c,
      out_channels = 2*c,
      kernel_size  = 7)
    # C2: [B x c*2 x (N-10-(7//2)*2)]
    # C2: [B x c*2 x (N-16)]
    self.conv3 = nn.Conv1d (
      in_channels  = 2*c,
      out_channels = 4*c,
      kernel_size  = 3)
    # C3: [B x c*4 x (N-16-(3//2)*2)]
    # C3: [B x c*4 x (N-18)]
    self.fc1 = nn.Linear(
      in_features=c*4*(n_scans-18),
      out_features=latent_dim)
    # OUT: [B x Latent Dim]
  def forward(self, x):
    c1 = self.conv1(x)
    c2 = self.conv2(c1)
    c3 = self.conv3(c2)
    c3 = c3.view(c3.size(0), -1)
    out = self.fc1(c3)
    out = torch.tanh(out)
    return out


class LidarDecoder(nn.Module):
  def __init__(self, capacity=8, 
            n_scans=90, latent_dim=10):
    super(LidarDecoder,self).__init__()
    c = capacity
    self.capacity = c
    self.n_scans  = n_scans
    self.fc1 = nn.Linear(
      in_features = latent_dim,
      out_features = c*4*(n_scans-18))
    self.c3 = nn.ConvTranspose1d(
      in_channels = c*4,
      out_channels = c*2,
      kernel_size = 3)
    self.c2 = nn.ConvTranspose1d(
      in_channels = c*2,
      out_channels = c,
      kernel_size = 7)
    self.c1 = nn.ConvTranspose1d(
      in_channels = c,
      out_channels = 1,
      kernel_size = 11)
  def forward(self, x):
    lin = self.fc1(x)
    lin = lin.view(x.size(0), self.capacity*4, self.n_scans - 18)
    c3  = self.c3 (lin)
    c2  = self.c2 (c3)
    c1  = self.c1 (c2)
    return c1


class LidarAutoencoder(nn.Module):
  def __init__(self, latent_dim=10, n_scans=90,
    capacity=12, x_min=0, x_max=5):
    super(LidarAutoencoder,self).__init__()
    self.encoder = LidarEncoder(capacity=capacity, 
      n_scans=n_scans, latent_dim=latent_dim)
    self.decoder = LidarDecoder(capacity=capacity, 
      n_scans=n_scans, latent_dim=latent_dim)
    self.x_min = x_min
    self.x_max = x_max
    
    self.apply(weights_init_2) 
    
  def forward(self, x):
    x = torch.Tensor(x).to(device)
    latent  = self.encoder(x)
    decoded = self.decoder(latent)
    # Limit to positive numbers
    decoded = torch.clamp(decoded,self.x_min,self.x_max) #F.relu(decoded)
    return decoded, latent
  
def visReconLidarScans(scan, robot_obj, col=[1,0,0]):
  scan_l1, scan_h1 = robot_obj.getLidarScanLimits(robot_obj.lidar_front_link)
  scan_l2, scan_h2 = robot_obj.getLidarScanLimits(robot_obj.lidar_rear_link)
  scan_l = np.concatenate((scan_l1, scan_l2))
  scan_h = np.concatenate((scan_h1, scan_h2))

  #scan = np.concatenate((scan[0],scan[1]))
  for i in range (len(scan)):
    hit = scan[i]*(scan_h[i] - scan_l[i])/robot_obj.scan_range + scan_l[i]
    p.addUserDebugLine(scan_l[i],
                       hit,
                       col)
  
  return

      

### DEFINE OBSTACLE FUNCITONS

def remove_all_obstacles_by_id(obstacle_ids):
  """
  Removes all obstacles from simulation
  """
  for entry in obstacle_ids:
    p.removeBody(entry)
  return

def generate_random_obstacles(n_obs=25):
  """
  Generate obstacles in random locations
  """
  obs_ids = []
  for _ in range (n_obs):
    angle = np.random.random()*np.pi*2
    x = np.sin(angle)
    y = np.cos(angle)
    r = np.random.random()*4 + 1
    rot = np.random.random()*np.pi*2
    wall = pb_Wall(pos = [x*r,y*r,0], 
               ori = [0,0,rot],
               size= [0.1,1,1])
    obs_ids.append(wall.id)
  return obs_ids
