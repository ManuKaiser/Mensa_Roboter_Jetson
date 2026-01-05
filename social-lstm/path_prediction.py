
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from rtSequenceBuffer import RealTimeSequenceBuffer
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_method_name, get_args
import copy

args = get_args()

tracker = DeepSort(max_age=30)

buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)

# Load LSTM
model, saved_args = load_model(args.method, "LSTM", args.epoch)

def prediction_step(ped_data, frame):

    results = []

    buffer.update(ped_data)

    if buffer.is_ready():
        x_seq, pedsList_seq, current_ids = buffer.get_sequence()

        # print("x_seq:", x_seq)
        # print("pedsList_seq:", pedsList_seq)
        # print("current_ids:", current_ids)
        # Prepare the sequence for prediction, use the ids from the current frame
        for target_id in current_ids:
            x_seq_copy = copy.deepcopy(x_seq)
            pedsList_seq_copy = copy.deepcopy(pedsList_seq)

            obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq_out, lookup_seq, first_values_dict = prepare_sequence(
                x_seq_copy, pedsList_seq_copy, saved_args, frame.shape, target_id
            )

            # print("Target ID:", target_id)
            # print("First frame IDs:", x_seq_copy[0][:, 0])
            # print("First frame coords:", x_seq_copy[0][:, 1:])



            # Run the model
            ret_x_seq = predict_trajectory(
                model, obs_traj, obs_PedsList_seq, args,
                pedsList_seq_copy, saved_args, frame.shape, lookup_seq,
                first_values_dict, use_gru=args.gru, obs_grid=obs_grid
            )

            predicted_array = [p[lookup_seq[target_id]].tolist() for p in ret_x_seq[-13:]]

            results.append((target_id, predicted_array))

    return results

