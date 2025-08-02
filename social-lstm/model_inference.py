import torch
import numpy as np
import pickle
import os
from helper import *
from grid import getSequenceGridMask
from test import sample
from torch.autograd import Variable
import pandas as pd

def load_model(method, model_name, epoch, base_dir=None):
    method_name = get_method_name(method)
    save_tar_name = method_name + "_lstm_model_"
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    save_directory = os.path.join(base_dir, 'model', method_name, model_name)
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    net = get_model(method, saved_args, True)
    if saved_args.use_cuda:
        net = net.cuda()

    checkpoint_path = os.path.join(save_directory, save_tar_name + str(epoch) + '.tar')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])

    return net, saved_args


def prepare_sequence(x_seq, pedsList_seq, saved_args, dimensions, target_id):
    from helper import clean_test_data, clean_ped_list, convert_proper_array, vectorize_seq

    obs_length = saved_args.seq_length - saved_args.pred_length
    seq_length = saved_args.seq_length

    clean_test_data(x_seq, target_id, obs_length, saved_args.pred_length)
    clean_ped_list(x_seq, pedsList_seq, seq_length)
    x_seq, lookup_seq = convert_proper_array(x_seq, pedsList_seq, seq_length=seq_length)
    grid_seq = getSequenceGridMask(x_seq, dimensions, pedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)
    x_seq, first_values_dict = vectorize_seq(x_seq, pedsList_seq, lookup_seq)
    x_seq = x_seq.cuda()

    obs_traj = x_seq[:obs_length]
    obs_PedsList_seq = pedsList_seq[:obs_length]
    obs_grid = grid_seq[:obs_length]

    return obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq, lookup_seq, first_values_dict


def predict_trajectory(model, obs_traj, obs_PedsList_seq, args, pedsList_seq, saved_args, dimensions, lookup_seq, first_values_dict, use_gru, obs_grid):
    from helper import revert_seq

    ret_x_seq = sample(obs_traj, obs_PedsList_seq, args, model, pedsList_seq, saved_args, dimensions, lookup_seq, use_gru, obs_grid)
    ret_x_seq = revert_seq(ret_x_seq, pedsList_seq, lookup_seq, first_values_dict)
    return ret_x_seq
