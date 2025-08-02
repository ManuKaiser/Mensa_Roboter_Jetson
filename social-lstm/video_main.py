
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from videostream import VideoStream
from rtSequenceBuffer import RealTimeSequenceBuffer
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_method_name, get_args

args = get_args()

# Load YOLO
yolo_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
cap = VideoStream(src=0)

buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)

# Load LSTM
model, saved_args = load_model(args.method, "LSTM", args.epoch)

while True:
    # Get frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model(frame)
    detections = []

    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf))

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
        ped_data.append([track_id, x_center / 10, y_center / 10])

        # Drawthe current bounding box and ID on the frame
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

    buffer.update(ped_data)

    if buffer.is_ready():
        x_seq, pedsList_seq, oldest_ids = buffer.get_sequence()
        # Prepare the sequence for prediction, use the ids from the oldest frames
        for target_id in oldest_ids:
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
            # scale the coordinates back to the original frame size
            pred_point = (int(x * 10), int(y * 10))

            # Get the current point of the target pedestrian
            for track in tracks:
                if track.track_id == target_id:
                    ltrb = track.to_ltrb()
                    current_point = (int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2))
                    break
            else:
                current_point = pred_point

            cv2.arrowedLine(frame, current_point, pred_point, (255, 0, 0), 5, tipLength=1)

    cv2.imshow("YOLO + Social LSTM", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
