import torch
import numpy as np
from model_inference import load_model, prepare_sequence, predict_trajectory
from helper import get_args
import cv2

# Dummy parameters
obs_length = 8
pred_length = 12
seq_length = obs_length + pred_length
person_id = 1

# Generate dummy straight-line trajectory
def generate_dummy_data(start=(100, 200), velocity=(5, 0), seq_length=20):
    x, y = start
    vx, vy = velocity
    x_seq = []
    pedsList_seq = []

    for i in range(seq_length):
        # Shape: (1, 3) = [ID, x, y]
        frame_data = np.array([[person_id, x / 10, y / 10]])  # normalize same as real data
        x_seq.append(frame_data)
        pedsList_seq.append(np.array([person_id]))  # list of IDs in this frame
        x += vx
        y += vy

    return x_seq, pedsList_seq

# --- Load model ---
args = get_args()
model, saved_args = load_model(args.method, "LSTM", args.epoch)

# --- Generate dummy sequence ---
x_seq, pedsList_seq = generate_dummy_data(seq_length=seq_length)

# --- Prepare and predict ---
frame_shape = (480, 640, 3)  # dummy frame size (height, width, channels)

obs_traj, obs_PedsList_seq, obs_grid, pedsList_seq, lookup_seq, first_values_dict = prepare_sequence(
    x_seq, pedsList_seq, saved_args, args, frame_shape, target_id=person_id
)

ret_x_seq = predict_trajectory(
    model, obs_traj, obs_PedsList_seq, args,
    pedsList_seq, saved_args, frame_shape, lookup_seq,
    first_values_dict, use_gru=args.gru, obs_grid=obs_grid
)

# --- Print output ---
print("Predicted trajectory (last 12 frames):")
for frame in ret_x_seq[obs_length:]:
    if frame is not None:
        print("x:", round(frame[0][0].item() * 10, 2), "y:", round(frame[0][1].item() * 10, 2))  # re-scale

# Optional: Visualize prediction
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
obs_points = [((x[0][0] * 10).item(), (x[0][1] * 10).item()) for x in ret_x_seq[:obs_length]]
pred_points = [((x[0][0] * 10).item(), (x[0][1] * 10).item()) for x in ret_x_seq[obs_length:]]

for pt in obs_points:
    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
for pt in pred_points:
    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
for i in range(1, len(pred_points)):
    cv2.line(canvas, (int(pred_points[i - 1][0]), int(pred_points[i - 1][1])),
             (int(pred_points[i][0]), int(pred_points[i][1])), (255, 0, 0), 2)

cv2.imshow("Prediction", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
