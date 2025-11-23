
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
from multiprocessing import shared_memory
import struct
from shm_writer import ShmPredictionWriter



args = get_args()

# Load YOLO
yolo_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
buffer = RealTimeSequenceBuffer(seq_length=args.obs_length + args.pred_length, observation_length=args.obs_length)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

align_to = rs.stream.color
align = rs.align(align_to)

profile = pipeline.start(config)

# Get the color stream intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

y_floor = 0.0  # Hight of the floor in meters, relativ to the camera, remember direction of the y axis

writer = ShmPredictionWriter(
    name="predictions_shm",
    max_targets=10,
    max_points_per_target=15
)

shm_image = shared_memory.SharedMemory(name='mensa_robot_shared_memory_image', create=True, size=1280*720*3)




# Load LSTM
model, saved_args = load_model(args.method, "LSTM", args.epoch)

try:
    while True:
        # Get frame from the video stream
        frameset = pipeline.wait_for_frames()
        aligned_frames = align.process(frameset)
        color_frame = aligned_frames.get_color_frame()
        depthframe = aligned_frames.get_depth_frame()

        frame = np.asanyarray(color_frame.get_data())
        
        # Run YOLO detection
        results = yolo_model(frame, verbose=False)
        detections = []

        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > 0.6:
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

            if not (0 <= x_person < intr.width and 0 <= y_person < intr.height):
                continue

            x, y, z = pixel_to_3dpoint(u=x_person, v=y_person, intr=intr, depth_frame=depthframe)
            if (x, y, z) == (0.0, 0.0, 0.0):    
                continue  # Skip invalid points
            ped_data.append([track_id, x, z])

            print(f"ID {track_id}: Pixel ({x_person}, {y_person}) -> 3D point ({x:.2f}, {y:.2f}, {z:.2f})")

            # Drawthe current bounding box and ID on the frame
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (x_person, y_person), 5, (0, 0, 255), -1)        

        results = prediction_step(ped_data, frame)
        # print(f"Prediction results: {results}")
        predictions = []

        for target_id, predicted_array in results:

            converted_pixel_points = []
            real_world_points = []

            for p in predicted_array:
                p3d = (p[0], y_floor, p[1])
                real_world_points.append(p3d)
                px = rs.rs2_project_point_to_pixel(intr, p3d)

                if px is None:
                    continue
                
                if len(px) != 2:
                    continue

                if not np.isfinite(px).all():
                    continue

                converted_pixel_points.append((int(px[0]), int(px[1])))

            # If we didn’t get at least two points, skip drawing + prediction entry
            if len(converted_pixel_points) < 2:
                continue

            predictions.append({
                "target_id": target_id,
                "points": real_world_points
            })


            current_point = converted_pixel_points[0]
            pred_point = converted_pixel_points[-1]


            # Only draw if both points are within the image
            if (0 <= current_point[0] < intr.width and 0 <= current_point[1] < intr.height and
                0 <= pred_point[0] < intr.width and 0 <= pred_point[1] < intr.height):
                cv2.arrowedLine(frame, current_point, pred_point, (0, 255, 255), 2, tipLength=0.5)
                print(f"Drawing arrow from {current_point} to {pred_point} for ID {target_id}")
                

        shm_image.buf[:frame.size] = frame.tobytes()
        writer.write(predictions)
        cv2.imshow("YOLO + Social LSTM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    shm_image.close()
    shm_image.unlink()

    writer.close()
    writer.unlink()