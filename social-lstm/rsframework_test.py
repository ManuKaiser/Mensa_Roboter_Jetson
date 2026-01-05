import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

started = False

try:
    profile = pipeline.start(config)
    started = True

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    print("Color camera intrinsics:")
    print(f"  Resolution: {intr.width} x {intr.height}")
    print(f"  fx: {intr.fx}")
    print(f"  fy: {intr.fy}")
    print(f"  ppx: {intr.ppx}")
    print(f"  ppy: {intr.ppy}")
    print(f"  Distortion model: {intr.model}")
    print(f"  Coefficients: {intr.coeffs}")

except RuntimeError as e:
    print("RealSense error:", e)

finally:
    if started:
        pipeline.stop()
