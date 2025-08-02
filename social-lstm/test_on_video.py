import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from rtSequenceBuffer import RealTimeSequenceBuffer
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_args

# === Args and Models ===
args = get_args()
video_path = "data/V_Train_0000.mp4"  # Change this to your input video file

# Load models
yolo_model = YOLO("yolov8n.pt")  # or custom path
tracker = DeepSort(max_age=30)
model, saved_args = load_model(args.method, "LSTM", args.epoch)

# Video input
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video output (optional)
save_output = True
if save_output:
    out = cv2.VideoWriter("output_prediction.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Trajectory buffer
buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length,
                                observation_length=args.obs_length)


# === Processing loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame)
    detections = []

    conf_threshold = 0.6  # Adjust as needed

    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf > conf_threshold:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf))


    # Track people
    tracks = tracker.update_tracks(detections, frame=frame)
    ped_data = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x_center = int((ltrb[0] + ltrb[2]) / 2)
        y_center = int((ltrb[1] + ltrb[3]) / 2)
        ped_data.append([track_id, x_center / 10, y_center / 10])

        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

    buffer.update(ped_data)

    # Predict trajectory
    if buffer.is_ready():
        x_seq, pedsList_seq, oldest_ids = buffer.get_sequence()
        for target_id in oldest_ids:
            obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq, lookup_seq, first_values_dict = prepare_sequence(
                x_seq, pedsList_seq, saved_args, args, frame.shape, target_id
            )

            ret_x_seq = predict_trajectory(
                model, obs_traj, obs_PedsList_seq, args,
                pedsList_seq, saved_args, frame.shape, lookup_seq,
                first_values_dict, use_gru=args.gru, obs_grid=obs_grid
            )

            x, y = ret_x_seq[-1][0].tolist()
            pred_point = (int(x * 10), int(y * 10))

            for track in tracks:
                if track.track_id == target_id:
                    ltrb = track.to_ltrb()
                    current_point = (int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2))
                    break
            else:
                current_point = pred_point

            cv2.arrowedLine(frame, current_point, pred_point, (255, 0, 0), 5, tipLength=0.4)
            cv2.line(frame, current_point, pred_point, (255, 0, 0), 5)

    # Show and/or save
    cv2.imshow("Trajectory Prediction", frame)
    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
