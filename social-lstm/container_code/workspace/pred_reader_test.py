import struct
from multiprocessing import shared_memory
import time

# Must match the writer settings
MAX_TARGETS = 10
MAX_POINTS_PER_TARGET = 15

# Compute maximum size
MAX_BYTES = 4 + MAX_TARGETS * (4 + MAX_POINTS_PER_TARGET * 12)


def read_predictions():
    shm = shared_memory.SharedMemory(name="predictions_shm")
    buf = shm.buf
    offset = 0

    (num_targets,) = struct.unpack_from("i", buf, offset)
    offset += 4

    results = []

    for _ in range(num_targets):
        (n_points,) = struct.unpack_from("i", buf, offset)
        offset += 4

        pts = []
        for _ in range(n_points):
            x, y, z = struct.unpack_from("fff", buf, offset)
            offset += 12
            pts.append((x, y, z))

        results.append(pts)

    shm.close()
    return results



if __name__ == "__main__":
    print("Reading predictions from shared memory...")

    while True:
        preds = read_predictions()
        if preds is None:
            time.sleep(1)
            continue

        print("\n--- PREDICTIONS ---")
        print(f"Targets detected: {len(preds)}")

        for i, pts in enumerate(preds):
            print(f" Target {i}: {len(pts)} points")
            print(f"   Points: {pts}")

        time.sleep(0.5)
