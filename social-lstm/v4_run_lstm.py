import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from collections import defaultdict, deque
import argparse
import pickle
import threading
import pandas as pd
import time


from videostream import VideoStream
from rtSequenceBuffer import RealTimeSequenceBuffer 
from helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from helper import *
from grid import getSequenceGridMask, getGridMask
from test import sample


def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

f_prefix = '.'
seq_length = 20

result_directory = os.path.join(f_prefix, 'result/', 'SOCIALLSTM')
model_name = "LSTM"



def get_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--maxNumPeds', type=int, default=27)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_cuda', action='store_true', default=torch.cuda.is_available())

    # Observation and prediction lengths
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')

    # Model to load
    parser.add_argument('--epoch', type=int, default=24,
                        help='Epoch of model to be loaded')
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True: GRU cell, False: LSTM cell')

    # LSTM method type
    parser.add_argument('--method', type=int, default=1,
                        help='1 = social LSTM, 2 = obstacle LSTM, 3 = vanilla LSTM')

    args = parser.parse_args([])  # Use [] to avoid command-line input in notebooks/scripts
    return args


def clean_test_data(x_seq, target_id, obs_length, predicted_length):
    for frame_num in range(obs_length):
        frame = x_seq[frame_num]
        is_tensor = isinstance(frame, torch.Tensor)
        frame_np = frame.cpu().numpy() if is_tensor else frame

        if frame_np.shape[1] == 3:  # [id, x, y]
            nan_mask = np.isnan(frame_np[:, 1]) | np.isnan(frame_np[:, 2])
        elif frame_np.shape[1] == 2:  # [x, y]
            nan_mask = np.isnan(frame_np[:, 0]) | np.isnan(frame_np[:, 1])
        else:
            raise ValueError(f"Unexpected frame shape: {frame_np.shape}")

        try:
            if is_tensor:
                keep_mask = torch.from_numpy(~nan_mask).to(frame.device)
                x_seq[frame_num] = frame[keep_mask]
            else:
                x_seq[frame_num] = frame[~nan_mask]
        except Exception as e:
            print(f"[Warning] Could not clean frame {frame_num}: {e}")

    for frame_num in range(obs_length, obs_length + predicted_length):
        frame = x_seq[frame_num]
        is_tensor = isinstance(frame, torch.Tensor)
        frame_np = frame.cpu().numpy() if is_tensor else frame

        if frame.shape[1] >= 1:
            id_mask = frame_np[:, 0] == target_id
            try:
                if is_tensor:
                    x_seq[frame_num] = frame[torch.from_numpy(id_mask).to(frame.device)]
                else:
                    x_seq[frame_num] = frame[id_mask]
            except Exception as e:
                print(f"[Warning] Could not filter target_id in frame {frame_num}: {e}")




def clean_ped_list(x_seq, pedlist_seq, seq_length):
    # remove peds from pedlist after test cleaning
    for frame_num in range(seq_length):
        pedlist_seq.append(x_seq[frame_num][:, 0])

def convert_proper_array( x_seq, pedlist, seq_length):
    #converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
    #array indices using a lookup table for each sequence -> speed
    #output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)
    
    #get unique ids from sequence

    unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
    # create a lookup table which maps ped ids -> array indices
    lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

    seq_data = np.zeros(shape=(seq_length, len(lookup_table), 2))

    # create new structure of array
    for ind, frame in enumerate(x_seq):
        corr_index = [lookup_table[x] for x in frame[:, 0]]
        seq_data[ind, corr_index,:] = frame[:,1:3]

    return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

    return return_arr, lookup_table

args = get_args()

# --- Load YOLOv8 model ---
yolo_model = YOLO("yolov8n.pt")

method_name = get_method_name(args.method)
model_name = "LSTM"
save_tar_name = method_name+"_lstm_model_"
base_dir = os.path.dirname(os.path.abspath(__file__))

# Save directory
save_directory = os.path.join(base_dir, 'model', method_name, model_name)

# Define the path for the config file for saved args
with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

pred_length = saved_args.pred_length
seq_length = saved_args.seq_length
obs_length = saved_args.seq_length - pred_length

net = get_model(args.method, saved_args, True)

if args.use_cuda:        
    net = net.cuda()

# Get the checkpoint path
checkpoint_path = os.path.join(save_directory, save_tar_name+str(args.epoch)+'.tar')
if os.path.isfile(checkpoint_path):
    print('Loading checkpoint')
    checkpoint = torch.load(checkpoint_path)
    model_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint at epoch', model_epoch)

# --- Init Deep SORT tracker ---
tracker = DeepSort(max_age=30)

# --- Video source ---
cap = VideoStream(src=0)
buffer = RealTimeSequenceBuffer(seq_length=seq_length, observation_length=obs_length)


# --- Parameters ---

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Run YOLOv8 inference
    results = yolo_model(frame)
    detections = []
    
    dimensions = frame.shape
    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # class 0 = person
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf))  # Deep SORT format

    # Step 2: Track with Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)

    ped_data = []
    # Step 3: Update history per tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x_center = int((ltrb[0] + ltrb[2]) / 2)
        y_center = int((ltrb[1] + ltrb[3]) / 2)
        ped_data.append([track_id, x_center/10, y_center/10])  
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
        
    # time.sleep(0.1)
    buffer.update(ped_data)

      

    if buffer.is_ready():

        x_seq, pedsList_seq, oldest_ids = buffer.get_sequence()
        for id in oldest_ids:
            target_id = id
            clean_test_data(x_seq, target_id, obs_length, pred_length)
            clean_ped_list(x_seq, pedsList_seq, seq_length)

            x_seq, lookup_seq = convert_proper_array(x_seq, pedsList_seq, seq_length)

            grid_seq = getSequenceGridMask(x_seq, dimensions, pedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

            x_seq, first_values_dict = vectorize_seq(x_seq, pedsList_seq, lookup_seq) # x_seq is now a list relative coordinates to the first values

            print(x_seq)
            print(first_values_dict)

            x_seq = x_seq.cuda()

            obs_traj, obs_PedsList_seq, obs_grid = x_seq[:obs_length], pedsList_seq[:obs_length], grid_seq[:obs_length]
            ret_x_seq = sample(obs_traj, obs_PedsList_seq, args, net, pedsList_seq, saved_args, dimensions, lookup_seq, args.gru, obs_grid)


            ret_x_seq = revert_seq(ret_x_seq, pedsList_seq, lookup_seq, first_values_dict)

            print("ret_x_seq", ret_x_seq)


            # The last position in the array, the furthest prediction
            x, y = ret_x_seq[19][0].tolist()
            print("x, y", x, y)

            # cv2.circle(frame, (int(x*10), int(y*10)), 5, (255, 0, 0), -1) # Draw predicted point

            # Scale prediction back up (you already scaled down by /10 earlier)
            pred_point = (int(x * 10), int(y * 10))

            # Find current point of this person
            for track in tracks:
                if track.track_id == target_id:
                    ltrb = track.to_ltrb()
                    current_point = (int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2))
                    break
            else:
                current_point = pred_point  # fallback

            # Draw arrow from current to predicted point
            cv2.arrowedLine(frame, current_point, pred_point, (255, 0, 0), 5, tipLength=1)

            # # Extract and scale predicted points
            # predicted_points = []
            # for point in ret_x_seq[obs_length:]:  # skip observed
            #     if point != None and isinstance(point, torch.Tensor) and point.numel() == 2:  # in case any are None
            #         x, y = point[0].tolist()
            #         predicted_points.append((int(x * 10), int(y * 10)))  # rescale to image space

            # # Draw lines between predicted points
            # for i in range(1, len(predicted_points)):
            #     cv2.line(frame, predicted_points[i - 1], predicted_points[i], color=(255, 0, 0), thickness=2)

            # # Optionally: draw circles at each predicted point
            # for pt in predicted_points:
            #     cv2.circle(frame, pt, radius=3, color=(0, 255, 255), thickness=-1)




    cv2.imshow("YOLO + Social LSTM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()