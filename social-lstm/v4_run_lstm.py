import numpy as np
from multiprocessing import shared_memory
import time

# Example: create a fake image (e.g. 640x480 RGB)
image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

# Create shared memory block large enough for the image
shm = shared_memory.SharedMemory(name="image_shm", create=True, size=image.nbytes)


# Keep the shared memory alive so container can access it
try:
    while True:
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        # Attach a NumPy array to the shared memory
        shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)

        # Write data into shared memory
        shared_image[:] = image
        time.sleep(1)
except KeyboardInterrupt:
    shm.close()
    shm.unlink()
