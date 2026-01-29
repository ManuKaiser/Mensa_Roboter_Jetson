import pyrealsense2 as rs
import numpy as np

def pixel_to_3dpoint(u=None, v=None, depth=None, intr=None, depth_frame=None):

    """
    Calculates the 3D point corresponding to a given pixel coordinate (u, v)
    in a depth frame.

    Parameters
    ----------
    u : int
    v : int
    depth : float, optional
        Depth value at the given pixel coordinate. Must be given if depth_frame is None.
    intr : rs.intrinsics, optional
        Intrinsics of the depth frame. Must be given if depth_frame is None.
    depth_frame : rs.depth_frame, optional
        Depth frame containing the depth value at the given pixel coordinate.

    Returns
    -------
    point : numpy.array
        3D point [X, Y, Z] corresponding to the given pixel coordinate.

    Raises
    -------
    ValueError
        If neither depth+intr nor depth_frame is given, or if the pixel coordinates (u, v) are None.
    """
    if depth_frame is None and (depth is None or intr is None):
        raise ValueError("Entweder depth+intr oder depth_frame muss übergeben werden.")

    if intr is None:
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    if u is None or v is None:
        raise ValueError("Pixelkoordinaten (u, v) muss angegeben werden.")

    if depth is None:
        depth = depth_frame.get_distance(u, v)
    

    point = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
    return np.array(point)  # [X, Y, Z]


def pixel_to_3dpoint_median(u=None, v=None, intr=None, depth_frame=None, h=None, w=None, r=None):
    """
    Calculates the 3D point corresponding to a given pixel coordinate (u, v)
    in a depth frame by taking the median of all depth values within a
    specified radius (r) around the pixel coordinate.

    Parameters
    ----------
    u : int
    v : int
        Pixel coordinates.
    intr : rs.intrinsics, optional
        Intrinsics of the depth frame. Must be given if depth_frame is None.
    depth_frame : rs.depth_frame, optional
        Depth frame containing the depth value at the given pixel coordinate.
        Must be given if depth is None.
    h : int, optional
        Height of the depth frame. Must be given if depth_frame is None.
    w : int, optional
        Width of the depth frame. Must be given if depth_frame is None.
    r : int, optional
        Radius around the pixel coordinate to take the median depth value.

    Returns
    -------
    point : numpy.array
        3D point [X, Y, Z] corresponding to the given pixel coordinate.

    Raises
    -------
    ValueError
        If neither depth+intr nor depth_frame is given, or if the pixel coordinates (u, v) are None.
    """
    if depth_frame is None and (depth is None or intr is None):
        raise ValueError("depth+intr or depth_frame must be given")

    if intr is None:
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    if u is None or v is None:
        raise ValueError("Pixelcoords(u, v) must be given")

    depths = []

    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            uu = u + du
            vv = v + dv

            # Bounds check
            if 0 <= uu < w and 0 <= vv < h:
                d = depth_frame.get_distance(uu, vv)
                if d > 0:  # ignore invalid depth
                    depths.append(d)

    if len(depths) == 0:
        return (0.0, 0.0, 0.0)

    depth = np.median(depths)
    
    point = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
    return np.array(point)  # [X, Y, Z]