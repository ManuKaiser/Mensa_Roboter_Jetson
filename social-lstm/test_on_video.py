
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
# from videostream import VideoStream
from rtSequenceBuffer import RealTimeSequenceBuffer
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_method_name, get_args
from path_prediction import prediction_step
import pyrealsense2 as rs
from pixel_to_3dpoint import pixel_to_3dpoint

args = get_args()

# Load YOLO
yolo_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)

# Get the color stream intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

z_floor = -1.0  # Hight of the floor in meters, relativ to the camera


# Load LSTM
model, saved_args = load_model(args.method, "LSTM", args.epoch)

while True:
    # Get frame from the video stream
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depthframe = frameset.get_depth_frame()

    frame = np.asanyarray(color_frame.get_data())
    
    # Run YOLO detection
    results = yolo_model(frame, verbose=False)
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
        x_person = int((ltrb[0] + ltrb[2]) / 2)
        y_person = int((ltrb[1] + ltrb[3]) / 2)

        x, y, z = pixel_to_3dpoint(u=x_person, v=y_person, intr=intr, depth_frame=depthframe)
        ped_data.append([track_id, x, z])

        print(f"ID {track_id}: Pixel ({x_person}, {y_person}) -> 3D point ({x:.2f}, {y:.2f}, {z:.2f})")