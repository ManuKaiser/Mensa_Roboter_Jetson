import cv2
import torch
import numpy as np
import time
import struct
import pyrealsense2 as rs
import signal
import threading
from multiprocessing import shared_memory

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from social_lstm.utils.rtSequenceBuffer import RealTimeSequenceBuffer
from social_lstm.utils.helper import get_args
from social_lstm.utils.path_prediction import prediction_step
from social_lstm.utils.pixel_to_3dpoint import pixel_to_3dpoint_median
from social_lstm.utils.shm_writer import ShmPredictionWriter



shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    """
    Signal handler for SIGINT and SIGTERM. Sets the shutdown event when a signal is received.
    """
    print(f"\nReceived signal {signum}, shutting down cleanly...")
    shutdown_event.set()


signal.signal(signal.SIGINT, handle_shutdown)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_shutdown)  # kill / systemd


# =========================
# Configuration Constants
# =========================
TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
Y_FLOOR = 0.4  # meters


# =========================
# Initialization Functions
# =========================
def init_models(args):
    """
    Initialize the YOLO model, DeepSort tracker and RealTimeSequenceBuffer for the video pipeline.

    Parameters:
    args (argparse.Namespace): Command line arguments

    Returns:
    tuple: A tuple containing the YOLO model, DeepSort tracker and RealTimeSequenceBuffer
    """
    yolo_model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=20)
    buffer = RealTimeSequenceBuffer(
        seq_length=args.obs_length + args.pred_length,
        observation_length=args.obs_length
    )
    return yolo_model, tracker, buffer


def init_realsense():
    """
    Initialize the RealSense pipeline and align object.

    The pipeline is configured to capture color and depth frames
    at FRAME_WIDTH x FRAME_HEIGHT resolution and 30 frames per second.

    The align object is used to align frames from the color and depth streams.

    The function returns the pipeline, align object and the intrinsics for the color stream.

    Returns:
        tuple: A tuple containing the pipeline, align object and intrinsics for the color stream
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    return pipeline, align, intrinsics


def init_shared_memory():
    """
    Initialize a shared memory segment to store the current frame and a ShmPredictionWriter
    to write the predictions to shared memory.

    Returns:
        tuple: A tuple containing the shared memory segment and the ShmPredictionWriter
    """
    shm_image = shared_memory.SharedMemory(
        name="mensa_robot_shared_memory_image",
        create=True,
        size=FRAME_WIDTH * FRAME_HEIGHT * 3
    )

    writer = ShmPredictionWriter(
        name="predictions_shm",
        max_targets=10,
        max_points_per_target=15
    )

    return shm_image, writer


# =========================
# Processing Functions
# =========================
def run_yolo(frame, yolo_model, conf_thresh=0.6):
    """
    Run YOLO on the given frame and return the detections.

    Parameters:
        frame (numpy.ndarray): The frame to run YOLO on.
        yolo_model (YOLO): The YOLO model to use.
        conf_thresh (float): The minimum confidence threshold for a detection to be considered valid.

    Returns:
        list: A list of tuples, where each tuple contains the bounding box coordinates and confidence of a detection.
    """
    detections = []
    results = yolo_model(frame, verbose=False)

    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf > conf_thresh:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf))

    return detections


def extract_pedestrians(tracks, depth_frame, intrinsics, frame):
    """
    Extract the pedestrian positions from the given tracks and depth frame.

    Parameters:
        tracks (list): A list of tracks from the DeepSort tracker.
        depth_frame (numpy.ndarray): The depth frame from the RealSense camera.
        intrinsics (rs.intrinsics): The intrinsics of the RealSense camera.
        frame (numpy.ndarray): The color frame from the RealSense camera.

    Returns:
        list: A list of tuples, where each tuple contains the track ID, x position, and z position of a pedestrian.
    """
    ped_data = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        x_center = int((ltrb[0] + ltrb[2]) / 2)
        y_center = int((ltrb[1] + ltrb[3]) / 2)

        if not (0 <= x_center < intrinsics.width and 0 <= y_center < intrinsics.height):
            continue

        x, y, z = pixel_to_3dpoint_median(
            u=x_center,
            v=y_center,
            intr=intrinsics,
            depth_frame=depth_frame,
            h=FRAME_HEIGHT,
            w=FRAME_WIDTH,
            r=10
        )

        if (x, y, z) == (0.0, 0.0, 0.0):
            continue

        ped_data.append([track_id, x, z])

        # Visualization
        cv2.rectangle(
            frame,
            (int(ltrb[0]), int(ltrb[1])),
            (int(ltrb[2]), int(ltrb[3])),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"ID {track_id}",
            (int(ltrb[0]), int(ltrb[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

    return ped_data


def draw_predictions(results, frame, intrinsics):
    """
    Draws 3D prediction results on a given frame.

    Args:
        results (list): List of tuples containing the target ID and the predicted 3D points.
        frame (numpy.ndarray): The frame to draw on.
        intrinsics (rs.intrinsics): Intrinsics of the camera.

    Returns:
        list: List of dictionaries containing the target ID and the predicted 3D points.
    """
    predictions = []

    for target_id, predicted_array in results:
        pixel_points = []
        real_world_points = []

        for p in predicted_array:
            p3d = (p[0], Y_FLOOR, p[1])
            real_world_points.append(p3d)

            px = rs.rs2_project_point_to_pixel(intrinsics, p3d)
            if px is None or len(px) != 2 or not np.isfinite(px).all():
                continue

            pixel_points.append((int(px[0]), int(px[1])))

        if len(pixel_points) < 2:
            continue

        predictions.append({
            "target_id": target_id,
            "points": real_world_points
        })

        base_color = np.array([0, 255, 255], dtype=np.float32)
        num_points = len(pixel_points)

        for i in range(num_points - 1):
            fade = 1 - (i / (num_points - 1))
            color = (base_color * fade).astype(int).tolist()

            p1, p2 = pixel_points[i], pixel_points[i + 1]

            if (
                0 <= p1[0] < intrinsics.width and
                0 <= p1[1] < intrinsics.height and
                0 <= p2[0] < intrinsics.width and
                0 <= p2[1] < intrinsics.height
            ):
                cv2.line(frame, p1, p2, color, 2)
                cv2.circle(frame, p1, 3, color, -1)

    return predictions


# =========================
# Main Loop
# =========================
def main():
    """
    Main loop for the video application.

    This function sets up the necessary objects (models, pipeline, shared memory)
    and runs an infinite loop where it processes frames from the pipeline,
    runs the YOLO model on the frames, updates the tracker, extracts pedestrian data,
    predicts future positions, draws the predictions on the frame and writes the
    frame and predictions to shared memory.

    Note: This function does not return anything. It's purpose is to run the video application.
    """
    args = get_args()
    yolo_model, tracker, buffer = init_models(args)
    pipeline, align, intrinsics = init_realsense()
    shm_image, writer = init_shared_memory()

    last_time = 0.0

    try:
        while not shutdown_event.is_set():
            frameset = pipeline.wait_for_frames()
            now = time.time()
            timestamp = time.time_ns()

            if now - last_time < FRAME_INTERVAL:
                continue
            last_time = now

            aligned = align.process(frameset)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            frame = np.asanyarray(color_frame.get_data())

            detections = run_yolo(frame, yolo_model)
            tracks = tracker.update_tracks(detections, frame=frame)

            ped_data = extract_pedestrians(tracks, depth_frame, intrinsics, frame)
            results = prediction_step(ped_data, frame)

            predictions = draw_predictions(results, frame, intrinsics)

            shm_image.buf[:frame.size] = frame.tobytes()
            writer.write(predictions, timestamp)

            cv2.imshow("Social LSTM", frame)
            cv2.waitKey(1)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        shm_image.close()
        shm_image.unlink()
        writer.close()
        writer.unlink()


if __name__ == "__main__":
    main()
