import numpy as np
from collections import deque

class RealTimeSequenceBuffer:
    def __init__(self, seq_length, observation_length):
        """
        Initialize a RealTimeSequenceBuffer object.

        Parameters:
        seq_length (int): The length of a sequence (number of frames).
        observation_length (int): The length of observations (number of frames to store in the buffer).

        Attributes:
        seq_length (int): The length of a sequence.
        observation_length (int): The length of observations.
        obs_buffer (deque): A buffer of observations (list of list of [id, x, y]).
        PedsList (deque): A buffer of pedestrian IDs per frame.
        """
        self.seq_length = seq_length
        self.observation_length = observation_length

        # Buffers for sequences
        self.obs_buffer = deque(maxlen=observation_length)       # List of list of [id, x, y]
        self.PedsList = deque(maxlen=observation_length)         # List of pedestrian IDs per frame

    def update(self, id_xy_list):
        """
        Update the sequence buffer with new observations.

        Parameters:
        id_xy_list (list of [id, x, y]): A list of observations.

        Returns:
        None
        """
        if len(id_xy_list) == 0:
            self.obs_buffer.append(np.empty((0, 3)))
            self.PedsList.append([])
            
            return

        arr = np.array(id_xy_list)  # shape: (num_peds, 3)
        self.obs_buffer.append(arr)
        self.PedsList.append(arr[:, 0].astype(int).tolist())
        

    def is_ready(self):
        """
        Check if the sequence buffer is ready for inference.

        Returns:
        bool: True if the sequence buffer is ready, False otherwise.
        """
        return len(self.obs_buffer) == self.observation_length

    def convert_x_seq_format(self, x_seq_str_arrays):
        """
        Convert a sequence of string arrays to a sequence of float arrays.

        Parameters:
        x_seq_str_arrays (list of list of str): A sequence of string arrays.

        Returns:
        list of numpy.ndarray: A sequence of float arrays.
        """
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
        Get the sequence of observations.

        Returns:
            x_seq (list of numpy.ndarray): A sequence of observations, where each frame is a numpy array of shape (num_peds, 3).
            pedsList_seq (list of list of int): A sequence of pedestrian IDs per frame.
            current_ids (list of int): A list of pedestrian IDs in the current frame.

        Notes:
            The sequence buffer will be cleared after calling this function.
        """
        x_seq_raw = list(self.obs_buffer)
        pedsList_seq = list(self.PedsList)

        if len(x_seq_raw) == 0:
            return [np.empty((0, 3)) for _ in range(self.seq_length)], [], []

        # Use IDs from the current available frame
        current_frame = x_seq_raw[self.observation_length - 1]
        current_ids = current_frame[:, 0].astype(int).tolist()

        # Align each frame to the current_ids
        aligned_seq = []
        for frame in x_seq_raw:
            id_to_coords = {int(row[0]): row[1:] for row in frame}
            frame_aligned = []
            for oid in current_ids:
                if oid in id_to_coords:
                    frame_aligned.append([oid, *id_to_coords[oid]])
                else:
                    frame_aligned.append([oid, np.nan, np.nan])
            aligned_seq.append(np.array(frame_aligned))

        # Pad the END if needed
        num_to_pad = self.seq_length - len(aligned_seq)
        if num_to_pad > 0:
            pad_frame = np.array([[oid, np.nan, np.nan] for oid in current_ids])
            aligned_seq += [pad_frame.copy() for _ in range(num_to_pad)]

        aligned_seq = self.convert_x_seq_format(aligned_seq)
        return aligned_seq, pedsList_seq, current_ids

    


