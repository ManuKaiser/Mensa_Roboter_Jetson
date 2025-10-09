import pyrealsense2 as rs
import numpy as np

def pixel_to_3dpoint(u=None, v=None, depth=None, intr=None, depth_frame=None):
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