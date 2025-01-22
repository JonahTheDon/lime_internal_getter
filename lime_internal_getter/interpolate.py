import numpy as np


def interp(x, x1, y1):
    """
    Perform linear interpolation with 32-bit precision.

    Parameters:
        x (float or array-like): The value(s) at which to interpolate.
        x1 (array-like): The x-coordinates of the data points.
        y1 (array-like): The y-coordinates of the data points.

    Returns:
        float or list: Interpolated value(s) corresponding to x.
    """
    # Convert inputs to numpy arrays with 32-bit precision
    x1 = np.array(x1, dtype=np.float32)
    y1 = np.array(y1, dtype=np.float32)

    # Handle single or multiple x values
    if np.isscalar(x):
        x = [x]
    x = np.array(x, dtype=np.float32)

    # Output list to store results
    interpolated_values = []

    for x_interp in x:
        # Find the index of the first x1 value greater than x_interp
        i = np.searchsorted(x1, x_interp)

        # Handle out-of-bounds conditions
        if i == 0:
            interpolated_values.append(float(y1[0]))  # Use the first y1 value
        elif i == len(x1):
            interpolated_values.append(float(y1[-1]))  # Use the last y1 value
        else:
            # Perform linear interpolation using 32-bit operations
            t = np.float32((x_interp - x1[i - 1]) / (x1[i] - x1[i - 1]))
            y_interp = y1[i - 1] + t * (y1[i] - y1[i - 1])
            interpolated_values.append(float(y_interp))

    # Return a single value if x was a scalar, otherwise return a list
    return (
        interpolated_values[0]
        if len(interpolated_values) == 1
        else interpolated_values
    )
