import numpy as np
from collections import deque

class RealTimeSequenceBuffer:
    def __init__(self, seq_length, observation_length):
        self.seq_length = seq_length
        self.observation_length = observation_length

        # Buffers for sequences
        self.obs_buffer = deque(maxlen=observation_length)       # List of list of [id, x, y]
        self.PedsList = deque(maxlen=observation_length)         # List of pedestrian IDs per frame

    def update(self, id_xy_list):
        """
        id_xy_list: list of [id, x, y] for each pedestrian in the frame
        """
        if len(id_xy_list) == 0:
            self.obs_buffer.append(np.empty((0, 3)))
            self.PedsList.append([])
            
            return

        arr = np.array(id_xy_list)  # shape: (num_peds, 3)
        self.obs_buffer.append(arr)
        self.PedsList.append(arr[:, 0].astype(int).tolist())
        

    def is_ready(self):
        return len(self.obs_buffer) == self.observation_length

    def convert_x_seq_format(self, x_seq_str_arrays):
        x_seq_float_arrays = []
        for frame in x_seq_str_arrays:
            float_frame = []
            for row in frame:
                try:
                    track_id = float(row[0])
                    x = float(row[1]) if row[1] != 'nan' else np.nan
                    y = float(row[2]) if row[2] != 'nan' else np.nan
                    float_frame.append([track_id, x, y])
                except (ValueError, IndexError):
                    continue  # Skip malformed entries
            x_seq_float_arrays.append(np.array(float_frame))
        return x_seq_float_arrays


    def get_sequence(self):
        """
        Returns:
            x_seq: list of np arrays, each with shape (num_oldest_ids, 3) -> [id, x, y]
                Length is always seq_length. Missing frames are padded at the end with NaNs.
            pedsList_seq: list of lists of IDs per actual frame
            oldest_ids: list of IDs from the oldest available frame
        """
        x_seq_raw = list(self.obs_buffer)
        pedsList_seq = list(self.PedsList)

        if len(x_seq_raw) == 0:
            return [np.empty((0, 3)) for _ in range(self.seq_length)], [], []

        # Use IDs from the oldest available frame
        oldest_frame = x_seq_raw[0]
        oldest_ids = oldest_frame[:, 0].astype(int).tolist()

        # Align each frame to the oldest_ids
        aligned_seq = []
        for frame in x_seq_raw:
            id_to_coords = {int(row[0]): row[1:] for row in frame}
            frame_aligned = []
            for oid in oldest_ids:
                if oid in id_to_coords:
                    frame_aligned.append([oid, *id_to_coords[oid]])
                else:
                    frame_aligned.append([oid, np.nan, np.nan])
            aligned_seq.append(np.array(frame_aligned))

        # Pad the END if needed
        num_to_pad = self.seq_length - len(aligned_seq)
        if num_to_pad > 0:
            pad_frame = np.array([[oid, np.nan, np.nan] for oid in oldest_ids])
            aligned_seq += [pad_frame.copy() for _ in range(num_to_pad)]

        aligned_seq = self.convert_x_seq_format(aligned_seq)
        return aligned_seq, pedsList_seq, oldest_ids

    


