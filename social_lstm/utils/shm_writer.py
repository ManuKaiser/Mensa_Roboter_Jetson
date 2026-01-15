import struct
import numpy as np
from multiprocessing import shared_memory

class ShmPredictionWriter:
    """
    Writes 3D prediction results to shared memory in a compact binary struct format.

    Correct layout (matching ROS reader):
        int32 num_targets
        For each target:
            int32 person_id
            int32 num_points
            repeated (float32 x, float32 y, float32 z)
    """

    def __init__(self, name="predictions_shm",
                 max_targets=10,
                 max_points_per_target=15):

        """
        Initialize a ShmPredictionWriter.

        Args:
            name (str, optional): Shared memory name. Defaults to "predictions_shm".
            max_targets (int, optional): Maximum number of targets to write. Defaults to 10.
            max_points_per_target (int, optional): Maximum number of points per target to write. Defaults to 15.
        """
        self.name = name
        self.max_targets = max_targets
        self.max_points_per_target = max_points_per_target

        self.max_bytes = 4 + 8 + max_targets * (4 + 4 + max_points_per_target * 12)

        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.max_bytes)
            self.existing = False
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            self.existing = True

        self.buf = self.shm.buf

    # ----------------------------------------------------------------------------------

    def write(self, predictions, timestamp):
        """
        predictions must be a list of dicts:
            {
                "target_id": int,
                "points": [(x,y,z), ...]
            }
        """

        offset = 0

        num_targets = min(len(predictions), self.max_targets)
        struct.pack_into("i", self.buf, offset, num_targets)
        offset += 4

        struct.pack_into("q", self.buf, offset, timestamp)
        offset += 8

        for i in range(num_targets):

            target_id = int(predictions[i]["target_id"])
            pts = predictions[i]["points"]

            # ------------------------------
            # 1. WRITE PERSON ID
            # ------------------------------
            struct.pack_into("i", self.buf, offset, target_id)
            offset += 4

            # ------------------------------
            # Filter & clean points
            # ------------------------------
            safe_pts = []
            for p in pts:
                if not hasattr(p, "__getitem__") or len(p) < 3:
                    continue

                x, y, z = float(p[0]), float(p[1]), float(p[2])

                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    continue

                safe_pts.append((x, y, z))
                if len(safe_pts) >= self.max_points_per_target:
                    break

            num_points = len(safe_pts)

            # ------------------------------
            # 2. WRITE num_points
            # ------------------------------
            struct.pack_into("i", self.buf, offset, num_points)
            offset += 4

            # ------------------------------
            # 3. WRITE POINT DATA
            # ------------------------------
            for (x, y, z) in safe_pts:

                # Safety check (prevent buffer overflow)
                if offset + 12 > self.max_bytes:
                    print("WARNING: SHM buffer overflow prevented.")
                    return

                struct.pack_into("fff", self.buf, offset, x, y, z)
                offset += 12

    # ----------------------------------------------------------------------------------

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
