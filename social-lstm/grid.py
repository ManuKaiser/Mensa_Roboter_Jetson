import numpy as np
import torch
import itertools
from torch.autograd import Variable


import numpy as np
import itertools

def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy=False):
    '''
    Computes binary occupancy masks for each pedestrian's neighborhood.
    '''
    mnp = num_person
    width, height = dimensions[0], dimensions[1]

    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size ** 2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size ** 2))

    frame_np = frame.data.numpy()
    width_bound = (neighborhood_size / (width * 1.0)) * 2
    height_bound = (neighborhood_size / (height * 1.0)) * 2

    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]
        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]

        # 🚨 Skip NaNs
        if np.isnan(current_x) or np.isnan(current_y) or np.isnan(other_x) or np.isnan(other_y):
            continue

        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2

        # If other person is not in the neighborhood, skip
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
            continue

        # Calculate grid cell
        cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
            continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y * grid_size] = 1
        else:
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y * grid_size] = 1

    return frame_mask


def getSequenceGridMask(sequence, dimensions, pedlist_seq, neighborhood_size, grid_size, using_cuda, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size, grid_size, is_occupancy)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask
