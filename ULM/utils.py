import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_ulm_tracks(img, MatTracking, raw_tracks, interpolated_tracks, save_path):
    frame_idx = img.shape[-1] // 2
    frame = np.abs(img[:, :, frame_idx])  # IQ frame background

    # 🔴 Combined Visualization: Localization and Tracks in{TH: 1 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    #
    # # Subplot 1: Localization
    ax1.imshow(frame, cmap='gray', extent=[0, frame.shape[1], frame.shape[0], 0])
    ax1.scatter(MatTracking[:, 2], MatTracking[:, 1], color='red', s=20, alpha=0.6)
    ax1.set_title("Localized Microbubbles")
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Subplot 2: Raw Tracks
    ax2.imshow(frame, cmap='gray', extent=[0, frame.shape[1], frame.shape[0], 0])
    for track in raw_tracks:
        if track.shape[0] >= 1:
            y, x = track[:, 0], track[:, 1]
            ax2.scatter(x, y, c='blue', s=50, alpha=0.7)
            ax2.plot(x, y, '-', color='blue', alpha=0.5)
    ax2.set_title("Raw Tracks")
    ax2.set_xlabel('X Position (pixels)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Subplot 3: Interpolated Tracks
    ax3.imshow(frame, cmap='gray', extent=[0, frame.shape[1], frame.shape[0], 0])
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
    plt.savefig(save_path)
    plt.show()