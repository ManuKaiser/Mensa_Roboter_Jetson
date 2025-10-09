import pyrealsense2 as rs
import numpy as np
import cv2

# --- RealSense pipeline setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Wait for a color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

# Get intrinsics
color_stream = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

mesh_image = color_image.copy()

# --- Y-plane ---
Y = 1.0
X_range = np.arange(-1.0, 1.05, 0.1)
Z_range = np.arange(0.5, 3.05, 0.1)
for x in X_range:
    prev = None
    for z in Z_range:
        pixel = rs.rs2_project_point_to_pixel(intr, [x, Y, z])
        u, v = int(pixel[0]), int(pixel[1])
        if 0 <= u < intr.width and 0 <= v < intr.height:
            cv2.circle(mesh_image, (u, v), 3, (255, 0, 0), -1)  # Blue
            if prev:
                cv2.line(mesh_image, prev, (u, v), (255, 0, 0), 1)
            prev = (u, v)
        else:
            prev = None

cv2.imwrite("grid_y_plane.png", mesh_image)
print("✅ Saved combined planes overlay: grid_y_plane.png")


pipeline.stop()
