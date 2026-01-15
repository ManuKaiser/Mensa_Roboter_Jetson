import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from utils import DataLoader
from grid import getSequenceGridMask, getGridMask
from helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from helper import *
from test import sample
import argparse

def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)


parser = argparse.ArgumentParser()
# Observed length of the trajectory parameter
parser.add_argument('--obs_length', type=int, default=8,
                    help='Observed length of the trajectory')
# Predicted length of the trajectory parameter
parser.add_argument('--pred_length', type=int, default=12,
                    help='Predicted length of the trajectory')


# Model to be loaded
parser.add_argument('--epoch', type=int, default=14,
                    help='Epoch of model to be loaded')
# cuda support
parser.add_argument('--use_cuda', action="store_true", default=True,
                    help='Use GPU or not')
# drive support
parser.add_argument('--drive', action="store_true", default=False,
                    help='Use Google drive or not')
# number of iteration -> we are trying many times to get lowest test error derived from observed part and prediction of observed
# part.Currently it is useless because we are using direct copy of observed part and no use of prediction.Test error will be 0.
parser.add_argument('--iteration', type=int, default=1,
                    help='Number of iteration to create test file (smallest test errror will be selected)')
# gru model
parser.add_argument('--gru', action="store_true", default=False,
                    help='True : GRU cell, False: LSTM cell')
# method selection
parser.add_argument('--method', type=int, default=1,
                    help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')

# Parse the parameters
sample_args = parser.parse_args()




f_prefix = '.'
model_name = "LSTM"
save_tar_name = 'SOCIALLSTM'+"_lstm_model_"
save_directory = os.path.join(f_prefix, 'model', 'SOCIALLSTM', model_name)

with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

obs_length = 8
pred_length = 12
seq_length = obs_length + pred_length

result_directory = os.path.join(f_prefix, 'result/', 'SOCIALLSTM')


dataloader = DataLoader(f_prefix, 1, seq_length, forcePreProcess = True, infer=True)

create_directories(os.path.join(result_directory, model_name), dataloader.get_all_directory_namelist())

dataloader.reset_batch_pointer()


net = get_model(sample_args.method, saved_args, True)

if sample_args.use_cuda:        
    net = net.cuda()

# Get the checkpoint path
checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
if os.path.isfile(checkpoint_path):
    print('Loading checkpoint')
    checkpoint = torch.load(checkpoint_path)
    model_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint at epoch', model_epoch)

x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch() 


print(x)

# print("x len:", len(x),
#       "y len:", len(y),
#       "d len:", len(d),
#       "PedsList len:", len(PedsList),
#       "target_ids len:", len(target_ids))
# x[0] = x[4]
# d[0] = d[4]
# PedsList[0] = PedsList[4]
# target_ids[0] = target_ids[4]



d_seq = d[0],
folder_name = dataloader.get_directory_name_with_pointer(d_seq[0])
dataset_data = dataloader.get_dataset_dimension(folder_name)

print("x[0]:", x[0])
print("d[0]:", d[0])
print("PedsList[0]:", PedsList[0])
print("target_ids[0]:", target_ids[0])


# #here

# print("----------------------------------------------------------------------------------------------------------")

# dataloader.clean_test_data(x[0], target_ids[0], obs_length, pred_length)
# dataloader.clean_ped_list(x[0], PedsList[0], target_ids[0], obs_length, pred_length)

# print("x[0] after cleaning:", x[0])

# print("PedsList[0]:", PedsList[0])
# print("target_ids[0]:", target_ids[0])

# print("----------------------------------------------------------------------------------------------------------")



# x_seq, lookup_seq = dataloader.convert_proper_array(x[0], PedsList[0])

# print("x_seq shape:", x_seq.shape)
# print("x_seq after convert_proper_array:")
# print(x_seq)

# orig_x_seq = x_seq.clone()


# grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList[0], saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

# print("-----------------------------------------------------------------------------------------------------------")
# print("grid_seq:")
# print(grid_seq)



# x_seq, first_values_dict = vectorize_seq(x_seq, PedsList[0], lookup_seq)

# print("x_seq after vectorization:")
# print(x_seq)
# print("first_values_dict:")
# print(first_values_dict)

# print("-----------------------------------------------------------------------------------------------------------")

# x_seq = x_seq.cuda()

# obs_traj, obs_PedsList_seq, obs_grid = x_seq[:obs_length], PedsList[0][:obs_length], grid_seq[:obs_length]
# ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, PedsList[0], saved_args, dataset_data, lookup_seq, sample_args.gru, obs_grid)

# print("-----------------------------------------------------------------------------------------------------------")
# print("ret_x_seq:") 
# print(ret_x_seq)

# ret_x_seq = revert_seq(ret_x_seq, PedsList[0], lookup_seq, first_values_dict)

# print("ret_x_seq after revert:")
# print(ret_x_seq)
# print("-----------------------------------------------------------------------------------------------------------")

# print(saved_args)