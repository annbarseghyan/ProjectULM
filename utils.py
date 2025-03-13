import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib.collections as collections
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_tracking_results(noisy_frames, MatTracking, raw_tracks, interpolated_tracks, plot_save_path):
    frame_idx = 1
    frame = np.abs(noisy_frames[:, :, frame_idx])
    frame_microbubbles = MatTracking[MatTracking[:, 3] == frame_idx]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(frame, cmap='gray', extent=[0, frame.shape[1], frame.shape[0], 0])
    ax1.scatter(frame_microbubbles[:, 2], frame_microbubbles[:, 1], color='red', s=30, alpha=0.6)
    ax1.set_title("Localized Microbubbles")
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    for track in raw_tracks:
        if track.shape[0] >= 1:
            y, x = track[:, 0], track[:, 1]
            ax2.scatter(x, y, c='blue', s=50, alpha=0.7)
            ax2.plot(x, y, '-', color='blue', alpha=0.5)
    ax2.set_title("Raw Tracks")
    ax2.set_xlabel('X Position (pixels)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    v_min = min([t[:, 2].min() for t in interpolated_tracks if t.size > 0]) if interpolated_tracks else 0
    v_max = max([t[:, 2].max() for t in interpolated_tracks if t.size > 0]) if interpolated_tracks else 1
    norm = plt.Normalize(v_min, v_max)

    for track in interpolated_tracks:
        if track.shape[0] >= 1:
            y, x, v_mean = track[:, 0], track[:, 1], track[:, 2]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = collections.LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(v_mean)
            lc.set_linewidth(2)
            ax3.add_collection(lc)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(lc, cax=cax, label='Mean Velocity (units/s)')

    ax3.set_title("Interpolated Tracks")
    ax3.set_xlabel('X Position (pixels)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

def load_mat_file(data_path):
    data = sio.loadmat(data_path)
    noisy_frames = data['IQs']
    clean_frames = data['GTs']
    noise_mat_frames = data['NoiseMats']
    return noisy_frames, clean_frames, noise_mat_frames


# Needed this function to make the distribution for the training images (deeploco simulation) and test images
# (Data chich embryo) similar
def match_image_range(source_img, target_img):
    """
    Adjust the pixel intensity range of source_img to match the range of target_img.
    """

    src_min, src_max = np.min(source_img), np.max(source_img)
    tgt_min, tgt_max = np.min(target_img), np.max(target_img)

    src_normalized = (source_img - src_min) / (src_max - src_min)

    matched_img = src_normalized * (tgt_max - tgt_min) + tgt_min

    return matched_img


def process_model_output(model, img, threshold=0.5):

    o_theta, o_w = model(img)
    points = o_theta.cpu().detach().numpy()
    probabilities = o_w.cpu().detach().numpy()

    results = []
    for frame_idx in range(points.shape[0]):
        frame_results = []
        mask = probabilities[frame_idx] > threshold
        selected_points = points[frame_idx][mask]
        for (x, y) in selected_points:
            frame_results.append((frame_idx, float(x)/100, float(y)/100))  # Ensure float format
            # Division by 100 as DeepLoco simulated images in nm
        results.append(frame_results)

    return results

def process_network_output(model, img, threshold=0.5):
    o_theta, o_w = model(img)

    points = o_theta.cpu().detach().numpy()
    weights = o_w.cpu().detach().numpy()

    # Filter points (simple thresholding, adjust as needed)
    batch_points = []
    batch_weights = []

    for b_idx in range(points.shape[0]):
        mask = weights[b_idx] > threshold
        filtered_points = points[b_idx, mask]
        filtered_weights = weights[b_idx, mask]
        batch_points.append(filtered_points)
        batch_weights.append(filtered_weights)

    return batch_points[0], batch_weights[0]


def visualize_deeploco_localization(simulated_data, real_data, save_figure_path):
    simulated_img, points_s, weights_s = simulated_data
    real_img, points_r, weights_r = real_data

    plt.ion()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(simulated_img[0].cpu(), extent=(0, 6400, 6400, 0), cmap='gray')
    axs[0].scatter(points_s[:, 0], points_s[:, 1], marker='x', color="red", s=weights_s * 100.0)
    axs[0].set_title("Localization on DeepLoco Simulation")
    axs[0].set_xlabel("x (nm)")
    axs[0].set_ylabel("y (nm)")

    axs[1].imshow(real_img[0].cpu(), extent=(0, 6400, 6400, 0), cmap='gray')
    axs[1].scatter(points_r[:, 0], points_r[:, 1], marker='x', color="red", s=weights_r * 100.0)
    axs[1].set_title("Localization on Real Data")
    axs[1].set_xlabel("x (nm)")
    axs[1].set_ylabel("y (nm)")

    plt.tight_layout()
    plt.savefig(save_figure_path)
    plt.show(block=True)
