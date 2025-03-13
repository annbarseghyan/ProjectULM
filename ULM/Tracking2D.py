import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline


def ULM_tracking2D_PALA(MatTracking, ULM):
    """
    Implements the PALA tracking method for ULM.

    Parameters:
        MatTracking (numpy.ndarray): [intensity, y, x, frame] matrix of microbubble positions.
        ULM (dict): Parameters:
            - 'max_linking_distance' (float): Maximum distance for linking microbubbles.
            - 'max_gap_closing' (int): Maximum frames allowed for gap closing.
            - 'min_length' (int): Minimum track length.
            - 'scale' (tuple): Scale factors [scale_z, scale_x, scale_t].
            - 'res' (float): Resolution factor.

    Returns:
        tuple: (raw_tracks, interpolated_tracks)
            - raw_tracks (list): List of raw tracks without interpolation.
            - interpolated_tracks (list): List of interpolated tracks with density rendering.
    """

    max_linking_distance = ULM['max_linking_distance']
    min_length = ULM['min_length']
    scale_t = ULM['scale'][2]
    interp_factor = 1 / max_linking_distance / ULM['res'] * 0.8

    # Adjust frame numbers to start from 1
    min_frame = int(np.min(MatTracking[:, 3]))
    MatTracking[:, 3] -= min_frame - 1
    num_frames = int(np.max(MatTracking[:, 3]))

    # Group bubbles by frame
    frame_bubbles = {i: MatTracking[MatTracking[:, 3] == i, 1:3] for i in range(1, num_frames + 1)}

    raw_tracks = []
    active_tracks = {}

    for t in range(1, num_frames):
        if t not in frame_bubbles or t + 1 not in frame_bubbles:
            continue

        prev_bubbles = frame_bubbles[t]
        next_bubbles = frame_bubbles[t + 1]

        if len(prev_bubbles) == 0 or len(next_bubbles) == 0:
            continue

        cost_matrix = cdist(prev_bubbles, next_bubbles)
        cost_matrix[cost_matrix > max_linking_distance] = np.inf

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
            print(f"Error: Cost matrix infeasible for frames {t} → {t + 1}. Skipping.")
            continue


        # Update active tracks
        new_active_tracks = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < max_linking_distance:
                # Find existing track or create a new one
                track_id = next((k for k, v in active_tracks.items() if np.array_equal(v[-1][:2], prev_bubbles[i])),
                                None)
                if track_id is None:
                    track_id = len(raw_tracks)
                    raw_tracks.append([])

                # Append new position
                raw_tracks[track_id].append([*next_bubbles[j], t + 1])
                new_active_tracks[track_id] = raw_tracks[track_id]

        active_tracks = new_active_tracks

    raw_tracks = [np.array(track) for track in raw_tracks if len(track) > min_length]

    interpolated_tracks = []
    for track in raw_tracks:
        if track.shape[0] < 2:  # Skip tracks with fewer than 2 points
            print(f"Skipping track with {track.shape[0]} points (too short for interpolation)")
            continue

        x = track[:, 1]
        y = track[:, 0]
        t = np.arange(len(track)) * scale_t

        n_interp_points = max(int(len(track) * interp_factor), len(track))
        t_interp = np.linspace(t[0], t[-1], n_interp_points)

        # Cubic spline interpolation
        if len(track) >= 4:
            spline_x = CubicSpline(t, x, bc_type='natural')
            spline_y = CubicSpline(t, y, bc_type='natural')
            x_interp = spline_x(t_interp)
            y_interp = spline_y(t_interp)
            # Velocity from derivatives
            dx_dt = spline_x(t_interp, 1)
            dy_dt = spline_y(t_interp, 1)
            v_mean = np.mean(np.sqrt(dx_dt ** 2 + dy_dt ** 2))
        else:
            # Linear interpolation for short tracks
            x_interp = np.interp(t_interp, t, x)
            y_interp = np.interp(t_interp, t, y)
            dd = np.sqrt(np.diff(x_interp) ** 2 + np.diff(y_interp) ** 2)
            v_mean = np.sum(dd) / (len(track) * scale_t) if len(dd) > 0 else 0.0


        interp_track = np.column_stack([y_interp, x_interp, v_mean * np.ones_like(y_interp)])
        interpolated_tracks.append(interp_track)

    return raw_tracks, interpolated_tracks





