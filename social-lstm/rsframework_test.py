import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)

# Get the color stream intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

print("Intrinsics:")
print("  width:", intr.width)
print("  height:", intr.height)
print("  fx:", intr.fx)
print("  fy:", intr.fy)
print("  ppx:", intr.ppx)
print("  ppy:", intr.ppy)
print("  distortion:", intr.model)
print("  coeffs:", intr.coeffs)


point_3d = [0.2, 0.1, 2.0]  # X=0.2m right, Y=0.1m down, Z=2.0m forward

pixel = rs.rs2_project_point_to_pixel(intr, point_3d)
u, v = int(pixel[0]), int(pixel[1])

print(f"Pixel coordinates: ({u}, {v})")


u, v = 640, 360

def pixel_to_camera(u=None, v=None, depth=None, intr=None, depth_frame=None):
    """
    Wandelt Pixelkoordinaten (u, v) + Tiefe in Kamerakoordinaten (X, Y, Z) um.

    Args:
        u, v (int): Pixelkoordinaten.
        depth (float): Tiefe in Metern an diesem Pixel.
        intr (rs.intrinsics): Kameraintrinsics.
        depth_frame (rs.depth_frame): Optionales Depth Frame, falls depth/intr fehlen.

    Returns:
        np.ndarray: [X, Y, Z] in Metern im Kamera-Koordinatensystem.
    """
    if depth_frame is None and (depth is None or intr is None):
        raise ValueError("Entweder depth+intr oder depth_frame muss übergeben werden.")

    # Falls Intrinsics fehlen, aus depth_frame holen
    if intr is None:
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    # Falls Tiefe fehlt, vom Depth Frame holen
    if depth is None:
        if u is None or v is None:
            raise ValueError("Pixelkoordinaten (u, v) müssen angegeben werden, wenn depth fehlt.")
        depth = depth_frame.get_distance(u, v)

    # Falls Pixelkoordinaten fehlen, nehme Bildmitte
    if u is None or v is None:
        u, v = intr.width // 2, intr.height // 2

    point = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
    return np.array(point)  # [X, Y, Z]


def camera_to_robot(point_camera, R=None, t=None):
    """
    Transformiert einen Punkt von Kamera- zu Roboterkoordinaten.

    Args:
        point_camera (np.ndarray): [X, Y, Z] im Kamera-Koordinatensystem.
        R (np.ndarray): 3x3 Rotationsmatrix Kamera->Roboter.
        t (np.ndarray): 3x1 Translation Kamera->Roboter.

    Returns:
        np.ndarray: [X, Y, Z] im Roboter-Koordinatensystem.
    """
    # Default: Kamera 20cm über Boden, gerade nach vorne
    if R is None:
        R = np.array([
            [0, 0, 1],   # Kamera-Z -> Roboter-X
            [-1, 0, 0],  # Kamera-X -> -Roboter-Y
            [0, -1, 0]   # Kamera-Y -> -Roboter-Z
        ])
    if t is None:
        t = np.array([0.0, 0.0, 0.2])  # Kamera 20cm über Boden

    point_robot = R @ point_camera + t
    return point_robot



while True:
    # This call waits until a new coherent set of frames is available on a device
    # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
    frames = pipeline.wait_for_frames()
    depthframe = frames.get_depth_frame()
    print(pixel_to_camera(u=u, v=v, depth=None, intr=intr, depth_frame=depthframe))

    