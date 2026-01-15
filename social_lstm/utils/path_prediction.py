
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from social_lstm.utils.rtSequenceBuffer import RealTimeSequenceBuffer
from social_lstm.utils.model_inference import load_model, prepare_sequence, predict_trajectory
from social_lstm.utils.helper import get_method_name, get_args
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

args = get_args()

tracker = DeepSort(max_age=30)

buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)

# Load LSTM
model, saved_args = load_model(
    args.method,
    "LSTM",
    args.epoch,
    model_dir=MODEL_DIR,
)


def prediction_step(ped_data, frame):

    """
    Make predictions for all pedestrians in the current frame.

    Parameters:
    ped_data (list of [id, x, y]): A list of pedestrian observations.
    frame (numpy array): The current frame.

    Returns:
    list of (id, predicted_array): A list of tuples containing the id of each pedestrian and an array of their predicted coordinates.
    """
    results = []

    buffer.update(ped_data)

    if buffer.is_ready():
        x_seq, pedsList_seq, current_ids = buffer.get_sequence()
        for target_id in current_ids:
            x_seq_copy = copy.deepcopy(x_seq)
            pedsList_seq_copy = copy.deepcopy(pedsList_seq)

            obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq_out, lookup_seq, first_values_dict = prepare_sequence(
                x_seq_copy, pedsList_seq_copy, saved_args, frame.shape, target_id
            )
            # Run the model
            ret_x_seq = predict_trajectory(
                model, obs_traj, obs_PedsList_seq, args,
                pedsList_seq_copy, saved_args, frame.shape, lookup_seq,
                first_values_dict, use_gru=args.gru, obs_grid=obs_grid
            )

            predicted_array = [p[lookup_seq[target_id]].tolist() for p in ret_x_seq[-13:]]

            results.append((target_id, predicted_array))

    return results

