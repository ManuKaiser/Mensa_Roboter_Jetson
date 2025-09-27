
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from videostream import VideoStream
from rtSequenceBuffer import RealTimeSequenceBuffer
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_method_name, get_args

args = get_args()

tracker = DeepSort(max_age=30)

buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)

# Load LSTM
model, saved_args = load_model(args.method, "LSTM", args.epoch)

def prediction_step(detections, frame):
    """
    detections in this format:
    detections = [
    [x, y, w, h, confidence, class_id],
    [x, y, w, h, confidence, class_id],
    ...
    ]
    frame: current video frame, need for DeepSort update
    Returns array of (id, current_point, pred_point) for each pedestrian 
    """
    results = []

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    ped_data = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x_center = int((ltrb[0] + ltrb[2]) / 2)
        y_center = int((ltrb[1] + ltrb[3]) / 2)
        # Collect pedestrian data and scale coordinates
        ped_data.append([track_id, x_center, y_center])

        # Drawthe current bounding box and ID on the frame
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

    buffer.update(ped_data)

    if buffer.is_ready():
        x_seq, pedsList_seq, current_ids = buffer.get_sequence()
        # Prepare the sequence for prediction, use the ids from the current frame
        for target_id in current_ids:
            obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq, lookup_seq, first_values_dict = prepare_sequence(
                x_seq, pedsList_seq, saved_args, frame.shape, target_id
            )

            # Run the model
            ret_x_seq = predict_trajectory(
                model, obs_traj, obs_PedsList_seq, args,
                pedsList_seq, saved_args, frame.shape, lookup_seq,
                first_values_dict, use_gru=args.gru, obs_grid=obs_grid
            )

            # The furthest predicted point
            x, y = ret_x_seq[-1][0].tolist()
            pred_point = (int(x), int(y))

            # The current point (last observed point)
            x, y = ret_x_seq[args.obs_length-1][0].tolist()
            current_point = (int(x), int(y))

            results.append((target_id, current_point, pred_point))

    return results


# Do you need the current point? Use the current point for each pedestrian after the inference has run, so the arrow starts at the correct position
# 2 threads, one for video capture, one for inference, updating the predicted point for each id, delete if id not in frame anymore