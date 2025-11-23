import numpy as np
from multiprocessing import shared_memory

WIDTH, HEIGHT = 640, 480
shape = (480, 640, 3)
dtype = np.uint8

# Attach to existing shared memory
shm = shared_memory.SharedMemory(name="image_shm")

# Create numpy array reading from shared memory
shared_image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

try:
    while True:
        img = shared_image.copy()  # copy to avoid race conditions
        # do something with img...
        print("Read image with mean pixel:", img.mean())
except KeyboardInterrupt:
    shm.close()
