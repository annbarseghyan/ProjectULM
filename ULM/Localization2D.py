import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.feature import peak_local_max
from scipy.optimize import least_squares


def ULM_localization2D(MatIn, ULM):
    """
    Perform 2D localization of microbubbles in ultrasound images.

    Parameters:
        MatIn (numpy.ndarray): 3D array of ultrasound images (height, width, frames).
        ULM (dict): Parameters for localization:
            - 'fwhm': Full Width at Half Maximum for bubble size (tuple: (fwhm_x, fwhm_z))
            - 'NLocalMax': Number of local maxima per frame
            - 'LocMethod': Localization method ('wa', 'interp', 'radial', 'curvefitting', 'nolocalization')
            - 'InterpMethod': Interpolation method ('bicubic', 'lanczos3', 'spline')
            - 'numberOfParticles': Estimated number of microbubbles per frame

    Returns:
        numpy.ndarray: Array containing detected microbubble intensities and positions.
    """
    fwhmz, fwhmx = ULM['fwhm']
    NLocalMax = ULM.get('NLocalMax', 3)
    LocMethod = ULM.get('LocMethod', 'radial')
    InterpMethod = ULM.get('InterpMethod', 'spline')
    num_particles = ULM.get('numberOfParticles', 10)

    height, width, num_frames = MatIn.shape
    MatIn = np.abs(MatIn)

    MatTracking = []

    # Define cropping mask range
    vectfwhmz = np.arange(-round(fwhmz / 2), round(fwhmz / 2) + 1)
    vectfwhmx = np.arange(-round(fwhmx / 2), round(fwhmx / 2) + 1)

    for t in range(num_frames):
        frame = MatIn[:, :, t]

        local_maxima = peak_local_max(frame, min_distance=round(fwhmx / 2), num_peaks=num_particles)
        if local_maxima.shape[0] == 0:
            continue

        intensities = frame[local_maxima[:, 0], local_maxima[:, 1]]
        sorted_indices = np.argsort(-intensities)[:NLocalMax]
        selected_maxima = local_maxima[sorted_indices]

        for y, x in selected_maxima:
            roi = frame[max(0, y - round(fwhmz / 2)): min(height, y + round(fwhmz / 2) + 1),
                  max(0, x - round(fwhmx / 2)): min(width, x + round(fwhmx / 2) + 1)]

            if roi.shape[0] < len(vectfwhmz) or roi.shape[1] < len(vectfwhmx):
                continue

            # Apply selected localization method
            if LocMethod == 'wa':
                dy, dx = LocWeightedAverage(roi, vectfwhmz, vectfwhmx)
            elif LocMethod == 'interp':
                dy, dx = LocInterp(roi, vectfwhmz, vectfwhmx)
            elif LocMethod == 'radial':
                dy, dx = LocRadialSym(roi)
            elif LocMethod == 'curvefitting':
                dy, dx = LocCurveFitting(roi)
            else:  # No localization
                dy, dx = 0, 0

            # Super-resolved position
            sub_y, sub_x = y + dy, x + dx

            # Store results
            MatTracking.append([frame[y, x], sub_y, sub_x, t])

    return np.array(MatTracking)


# **Localization Methods**
def LocWeightedAverage(Iin, vectfwhmz, vectfwhmx):
    """Weighted average localization."""
    sum_intensity = np.sum(Iin)
    dy = np.sum(Iin * vectfwhmz[:, None]) / sum_intensity
    dx = np.sum(Iin * vectfwhmx[None, :]) / sum_intensity
    return dy, dx


def LocInterp(Iin, vectfwhmz, vectfwhmx):
    """Interpolation-based localization."""
    Nz, Nx = Iin.shape
    x = np.linspace(0, Nx - 1, Nx)
    z = np.linspace(0, Nz - 1, Nz)

    interp_func = RectBivariateSpline(z, x, Iin, kx=3, ky=3)
    fine_x = np.linspace(0, Nx - 1, Nx * 10)
    fine_z = np.linspace(0, Nz - 1, Nz * 10)
    fine_grid = interp_func(fine_z, fine_x)

    max_idx = np.unravel_index(np.argmax(fine_grid), fine_grid.shape)
    dy = vectfwhmz[0] + max_idx[0] / 10 - 0.05
    dx = vectfwhmx[0] + max_idx[1] / 10 - 0.05
    return dy, dx


def LocRadialSym(Iin):
    """Radial symmetry-based localization."""
    Ny, Nx = Iin.shape
    Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
    center_y, center_x = Ny // 2, Nx // 2

    grad_y, grad_x = np.gradient(Iin)
    norm_grad = np.sqrt(grad_y ** 2 + grad_x ** 2)
    valid = norm_grad > 1e-6

    grad_y[~valid] = 0
    grad_x[~valid] = 0

    div_x = np.sum(X * grad_x) / np.sum(grad_x)
    div_y = np.sum(Y * grad_y) / np.sum(grad_y)

    return div_y - center_y, div_x - center_x


def LocCurveFitting(Iin):
    """Curve fitting localization using a 2D Gaussian model."""

    def gaussian_2d(params, x, y):
        x0, y0, sigma = params
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    Ny, Nx = Iin.shape
    Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')

    def residuals(params):
        return np.ravel(Iin - gaussian_2d(params, X, Y))

    initial_guess = [Ny / 2, Nx / 2, 1]
    result = least_squares(residuals, initial_guess)

    dy = result.x[0] - Ny // 2
    dx = result.x[1] - Nx // 2
    return dy, dx
