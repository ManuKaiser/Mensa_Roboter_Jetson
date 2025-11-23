import struct
import numpy as np
from multiprocessing import shared_memory

class ShmPredictionWriter:
    """
    Writes 3D prediction results to shared memory in a compact binary struct format.

    Layout:
        int32 num_targets
        For each target:
            int32 num_points
            repeated (float32 x, float32 y, float32 z)
    """

    def __init__(self, name="predictions_shm",
                 max_targets=20,
                 max_points_per_target=20):

        self.name = name
        self.max_targets = max_targets
        self.max_points_per_target = max_points_per_target

        # Compute maximum bytes needed:
        # num_targets (4)
        # per target: num_points (4) + points * 12 bytes
        self.max_bytes = (
            4 + max_targets * (4 + max_points_per_target * 12)
        )

        # Create or attach shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True,
                                                  size=self.max_bytes)
            self.existing = False
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            self.existing = True

        self.buf = self.shm.buf

    # ------------------------------------------------------------

    def write(self, predictions):
        """
        predictions must be a list of:
            {
                "target_id": ...,
                "points": [(x,y,z), (x,y,z), ...]
            }
        """

        offset = 0

        num_targets = min(len(predictions), self.max_targets)

        # Write number of targets
        struct.pack_into("i", self.buf, offset, num_targets)
        offset += 4

        for i in range(num_targets):

            pts = predictions[i]["points"]

            # ------------------------------------------------------
            # SAFE FILTERING FOR 3D POINTS
            # ------------------------------------------------------
            safe_pts = []
            for p in pts:

                # Must be indexable with ≥3 values
                if not hasattr(p, "__getitem__") or len(p) < 3:
                    continue

                x, y, z = p[0], p[1], p[2]

                # Must be finite numeric values
                if not (
                    np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
                ):
                    continue

                # Convert to float32 safely
                try:
                    xf = float(x)
                    yf = float(y)
                    zf = float(z)
                except Exception:
                    continue

                safe_pts.append((xf, yf, zf))

                if len(safe_pts) >= self.max_points_per_target:
                    break

            # ------------------------------------------------------
            # WRITE NUMBER OF SAFE POINTS
            # ------------------------------------------------------
            n = len(safe_pts)
            struct.pack_into("i", self.buf, offset, n)
            offset += 4

            # ------------------------------------------------------
            # WRITE POINTS (float32 x, y, z)
            # ------------------------------------------------------
            for (xf, yf, zf) in safe_pts:
                struct.pack_into("fff", self.buf, offset, xf, yf, zf)
                offset += 12

    # ------------------------------------------------------------

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
