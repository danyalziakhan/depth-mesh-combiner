from __future__ import annotations

import importlib
import cv2

import numpy as np

from scipy.ndimage import rotate


def stretch_top_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Vertically stretches (pulls outward) the top portion of a 2D image or depth map,
    while keeping the bottom portion unchanged.

    Applies a nonlinear transformation to the top side of the image by remapping pixel
    coordinates to stretch space vertically. The amount of stretch is controlled by
    the `strength` parameter. The region above the `split_ratio` remains unaffected.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Stretch intensity. Must be > 1. Higher values produce stronger stretching.
        split_ratio (float): Value in (0, 1). Vertical position (as a fraction of height)
                             where stretching of the top ends.

    Returns:
        np.ndarray: The resulting image or depth map with the top portion stretched outward.

    Example:
        result = stretch_top_only(depth_array, strength=2.5, split_ratio=0.4)
    """

    height, width = depth_array.shape
    y_indices = np.linspace(0, 1, height)

    y_mapped = np.copy(y_indices)
    top_mask = y_indices <= split_ratio
    top_y = y_indices[top_mask] / split_ratio  # normalize to [0,1]
    stretched = top_y ** (1 / strength)
    y_mapped[top_mask] = stretched * split_ratio

    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), y_mapped * (height - 1)
    )
    return cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shrink_bottom_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Vertically stretches (pulls outward) the bottom portion of a 2D image or depth map,
    while keeping the top portion unchanged.

    Despite the function name, it applies an outward stretching transformation to the
    bottom side of the image using a nonlinear remapping. Pixels below the `split_ratio`
    point are pushed further downward, depending on the `strength` parameter. The top portion
    remains unaffected.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Stretch intensity. Must be > 1. Higher values result in stronger stretching.
        split_ratio (float): Value in (0, 1). Vertical position (as a fraction of height)
                             where stretching of the bottom begins.

    Returns:
        np.ndarray: The resulting image or depth map with the bottom portion stretched outward.

    Example:
        result = shrink_bottom_only(depth_array, strength=2.0, split_ratio=0.6)
    """

    height, width = depth_array.shape
    y_indices = np.linspace(0, 1, height)

    y_mapped = np.copy(y_indices)
    bottom_mask = y_indices >= split_ratio
    bottom_y = (y_indices[bottom_mask] - split_ratio) / (1 - split_ratio)
    stretched = bottom_y ** (1 / strength)
    y_mapped[bottom_mask] = split_ratio + stretched * (1 - split_ratio)

    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), y_mapped * (height - 1)
    )
    return cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def stretch_left_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Horizontally stretches (pulls outward) the left portion of a 2D image or depth map,
    while keeping the right portion unchanged.

    This function remaps the left side of the image using a non-linear transformation that
    pushes pixels farther toward the left edge, effectively "stretching" it outward.
    The transformation is stronger with higher `strength`. The right side (from `split_ratio`
    to the right edge) remains unaffected.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Stretch intensity. Must be > 1. Higher values make the stretching more aggressive.
        split_ratio (float): Value in (0, 1). Horizontal position (as a fraction of width)
                             where stretching of the left ends and the unchanged right begins.

    Returns:
        np.ndarray: The resulting image or depth map with the left portion stretched outward.

    Example:
        result = stretch_left_only(depth_array, strength=2.5, split_ratio=0.4)
    """

    height, width = depth_array.shape
    x_indices = np.linspace(0, 1, width)

    x_mapped = np.copy(x_indices)
    left_mask = x_indices <= split_ratio
    left_x = x_indices[left_mask] / split_ratio
    stretched = left_x ** (1 / strength)
    x_mapped[left_mask] = stretched * split_ratio

    map_x, map_y = np.meshgrid(
        x_mapped * (width - 1), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def stretch_right_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Stretches the right portion of the image *outward* (to the right),
    keeping the left portion intact.

    Parameters:
        depth_array: 2D NumPy array (image or depth map)
        strength: float, >1 for more stretch
        split_ratio: float in (0,1), horizontal split point (left to right)
    Returns:
        Warped 2D array with the right portion stretched outward.
    """
    height, width = depth_array.shape
    x_indices = np.linspace(0, 1, width)

    x_mapped = np.copy(x_indices)

    right_mask = x_indices >= split_ratio
    right_x = (x_indices[right_mask] - split_ratio) / (
        1 - split_ratio
    )  # Normalize to [0,1]
    stretched = 1 - (1 - right_x) ** (1 / strength)  # Flipped outward stretch
    x_mapped[right_mask] = split_ratio + stretched * (1 - split_ratio)

    map_x, map_y = np.meshgrid(
        x_mapped * (width - 1), np.arange(height, dtype=np.float32)
    )

    return cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shrink_top_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Vertically compresses (shrinks) the top portion of a 2D image or depth map inward,
    toward the center, while keeping the bottom portion unchanged.

    This function remaps the top part of the image using a non-linear transformation that
    pulls pixels downward. The transformation is stronger with higher `strength`, resulting
    in a compressed or "inward-stretched" top. The bottom (from `split_ratio` to bottom edge)
    remains unaffected.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Compression intensity. Must be > 1. Larger values increase compression strength.
        split_ratio (float): Value in (0, 1). Vertical position (as a fraction of height) where
                             compression of the top ends and the unchanged bottom begins.

    Returns:
        np.ndarray: The resulting image or depth map with the top compressed inward.

    Example:
        result = shrink_top_only(depth_array, strength=3.0, split_ratio=0.4)
    """

    height, width = depth_array.shape
    y_indices = np.linspace(0, 1, height)

    # Compute 1D mapping for vertical axis
    y_mapped = np.copy(y_indices)
    top_mask = y_indices <= split_ratio
    top_y = y_indices[top_mask] / split_ratio  # normalize to [0,1]
    compressed = top_y**strength
    y_mapped[top_mask] = compressed * split_ratio

    # Create meshgrid
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), y_mapped * (height - 1)
    )

    # Convert to correct types
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def stretch_bottom_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Compresses the bottom portion of the image inward (toward the center), keeping the top intact.

    Parameters:
        depth_array: 2D NumPy array (e.g., depth image)
        strength: float, compression curve strength (>1 = more compressed)
        split_ratio: float in (0,1), vertical point below which compression applies
    Returns:
        Warped 2D array with compressed bottom
    """

    height, width = depth_array.shape
    y_indices = np.linspace(0, 1, height)

    y_mapped = np.copy(y_indices)
    mask = y_indices >= split_ratio
    y_normalized = (y_indices[mask] - split_ratio) / (1 - split_ratio)
    y_compressed = y_normalized**strength
    y_mapped[mask] = split_ratio + y_compressed * (1 - split_ratio)

    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), y_mapped * (height - 1)
    )
    return cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shrink_left_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Horizontally compresses (shrinks) the left portion of a 2D image or depth map inward,
    toward the center, while keeping the right portion unchanged.

    This function remaps the left side of the input array by applying a non-linear transformation
    that pulls pixels inward, creating a compressed effect. The right portion (from `split_ratio`
    to the end) remains unaltered.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Compression strength. Must be > 1. Higher values cause stronger inward pull.
        split_ratio (float): Value in (0, 1). Horizontal position (as a fraction of width) where
                             compression of the left portion ends.

    Returns:
        np.ndarray: The transformed 2D array with the left portion compressed inward.

    Example:
        compressed = shrink_left_only(depth_array, strength=2.5, split_ratio=0.4)
    """

    height, width = depth_array.shape
    x_indices = np.linspace(0, 1, width)

    # Compute 1D mapping for horizontal axis
    x_mapped = np.copy(x_indices)
    left_mask = x_indices <= split_ratio
    left_x = x_indices[left_mask] / split_ratio  # normalize to [0,1]
    compressed = left_x**strength
    x_mapped[left_mask] = compressed * split_ratio

    # Now convert 1D x_mapped into 2D grid using meshgrid
    map_x, map_y = np.meshgrid(
        x_mapped * (width - 1), np.arange(height, dtype=np.float32)
    )

    # Ensure correct types for remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shrink_right_only(
    depth_array: np.ndarray, strength: float = 2.0, split_ratio: float = 0.5
):
    """
    Horizontally compresses (shrinks) the right portion of a 2D image or depth map toward the center,
    while keeping the left portion unchanged.

    This function remaps the right side of the input array by applying a non-linear transformation
    that pulls pixels inward, creating a compressed visual effect. The left portion (from start
    up to `split_ratio`) remains unaltered.

    Parameters:
        depth_array (np.ndarray): A 2D NumPy array representing an image or depth map.
        strength (float): Compression strength. Must be > 1. Higher values cause stronger inward pull.
        split_ratio (float): Value in (0, 1). Horizontal position (as a fraction of width) from which
                             compression of the right portion begins.

    Returns:
        np.ndarray: The transformed 2D array with the right portion compressed inward.

    Example:
        compressed = shrink_right_only(depth_array, strength=2.5, split_ratio=0.6)
    """

    height, width = depth_array.shape
    x_indices = np.linspace(0, 1, width)

    # Compute 1D mapping for horizontal axis
    x_mapped = np.copy(x_indices)
    right_mask = x_indices >= split_ratio
    right_x = (x_indices[right_mask] - split_ratio) / (1 - split_ratio)
    compressed = 1 - (1 - right_x) ** strength
    x_mapped[right_mask] = split_ratio + compressed * (1 - split_ratio)

    # Now convert 1D x_mapped into 2D grid using meshgrid
    map_x, map_y = np.meshgrid(
        x_mapped * (width - 1), np.arange(height, dtype=np.float32)
    )

    # Ensure correct types for remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_depth_array_consistent(
    depth_array: np.ndarray,
    depth_array1_top: int,
    depth_array1_bottom: int,
    depth_array1_left: int,
    depth_array1_right: int,
    depth_array2_top: int,
    depth_array2_bottom: int,
    depth_array2_left: int,
    depth_array2_right: int,
    depth_array3_top: int,
    depth_array3_bottom: int,
    depth_array3_left: int,
    depth_array3_right: int,
    depth_array4_top: int,
    depth_array4_bottom: int,
    depth_array4_left: int,
    depth_array4_right: int,
    device_idx: int,
    target_width=456,  # Adjust based on your actual depth_array width
):
    """Crops a depth array according to device-specific boundaries and pads to a target width.

    This function selects a region of the input array based on the device index and provided crop boundaries, then pads the result to the specified width.

    Args:
        depth_array: 2D NumPy array to crop.
        depth_array1_top: Top boundary for device 1.
        depth_array1_bottom: Bottom boundary for device 1.
        depth_array1_left: Left boundary for device 1.
        depth_array1_right: Right boundary for device 1.
        depth_array2_top: Top boundary for device 2.
        depth_array2_bottom: Bottom boundary for device 2.
        depth_array2_left: Left boundary for device 2.
        depth_array2_right: Right boundary for device 2.
        depth_array3_top: Top boundary for device 3.
        depth_array3_bottom: Bottom boundary for device 3.
        depth_array3_left: Left boundary for device 3.
        depth_array3_right: Right boundary for device 3.
        depth_array4_top: Top boundary for device 4.
        depth_array4_bottom: Bottom boundary for device 4.
        depth_array4_left: Left boundary for device 4.
        depth_array4_right: Right boundary for device 4.
        device_idx: Index of the device (0-3) to determine which crop boundaries to use.
        target_width: Desired width of the output array after padding.

    Returns:
        Cropped and padded 2D NumPy array.
    """
    if device_idx == 0:  # depth_array 1 (top-left)
        arr = depth_array[
            depth_array1_top:depth_array1_bottom, depth_array1_left:depth_array1_right
        ]
        pad_left = target_width - arr.shape[1]
        if pad_left > 0:
            arr = np.pad(arr, ((0, 0), (pad_left, 0)), mode="constant")
    elif device_idx == 1:  # depth_array 2 (top-right)
        arr = depth_array[
            depth_array2_top:depth_array2_bottom, depth_array2_left:depth_array2_right
        ]
        pad_right = target_width - arr.shape[1]
        if pad_right > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_right)), mode="constant")
    elif device_idx == 2:  # depth_array 3 (bottom-left)
        arr = depth_array[
            depth_array3_top:depth_array3_bottom, depth_array3_left:depth_array3_right
        ]
        pad_left = target_width - arr.shape[1]
        if pad_left > 0:
            arr = np.pad(arr, ((0, 0), (pad_left, 0)), mode="constant")
    elif device_idx == 3:  # depth_array 4 (bottom-right)
        arr = depth_array[
            depth_array4_top:depth_array4_bottom, depth_array4_left:depth_array4_right
        ]
        pad_right = target_width - arr.shape[1]
        if pad_right > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_right)), mode="constant")

    return arr


def get_combined_array(depth_arrays: list[np.ndarray]):
    """Combines four depth arrays into a single 2D array arranged in a 2x2 grid.

    This function pads and concatenates the input arrays as needed to form a single array with the top row as [A | B] and the bottom row as [C | D].

    Args:
        depth_arrays: List of four 2D NumPy arrays to combine.

    Returns:
        A single 2D NumPy array with the input arrays arranged in a 2x2 grid.
    """
    # [ A | B ]
    # [ C | D ]

    diff_shapet = depth_arrays[1].shape[0] - depth_arrays[0].shape[0]

    # Pad depth_arrays[0] with zeros at the beginning along the first dimension
    if diff_shapet > 0:
        depth_arrays[0] = np.pad(
            depth_arrays[0], ((diff_shapet, 0), (0, 0)), mode="constant"
        )

    top_row = np.concatenate((depth_arrays[0], depth_arrays[1]), axis=1)

    diff_shapeb = depth_arrays[0].shape[0] - depth_arrays[1].shape[0]

    # Pad depth_arrays[1] with zeros at the end along the first dimension
    if diff_shapeb > 0:
        depth_arrays[1] = np.pad(
            depth_arrays[1], ((0, diff_shapeb), (0, 0)), mode="constant"
        )

    max_shape_bottom = max(depth_arrays[2].shape[0], depth_arrays[3].shape[0])

    # Pad depth_arrays[2] with zeros at the end along the first dimension
    depth_arrays[2] = np.pad(
        depth_arrays[2],
        ((0, max_shape_bottom - depth_arrays[2].shape[0]), (0, 0)),
        mode="constant",
    )

    # Pad depth_arrays[3] with zeros at the end along the first dimension
    depth_arrays[3] = np.pad(
        depth_arrays[3],
        ((0, max_shape_bottom - depth_arrays[3].shape[0]), (0, 0)),
        mode="constant",
    )

    # Concatenate depth_arrays[2] and depth_arrays[3] horizontally
    bottom_row = np.concatenate((depth_arrays[2], depth_arrays[3]), axis=1)

    diff_shape1 = bottom_row.shape[1] - top_row.shape[1]

    # Pad top_row with zeros at the beginning along the second dimension
    if diff_shape1 > 0:
        top_row = np.pad(top_row, ((0, 0), (diff_shape1, 0)), mode="constant")

    diff_shape = top_row.shape[1] - bottom_row.shape[1]
    if diff_shape > 0:
        bottom_row = np.pad(bottom_row, ((0, 0), (0, diff_shape)), mode="constant")

    return np.concatenate((top_row, bottom_row), axis=0)


def crop_depth_arrays_consistent(
    depth_array1: np.ndarray,
    depth_array2: np.ndarray,
    depth_array3: np.ndarray,
    depth_array4: np.ndarray,
    depth_array1_top: int,
    depth_array1_bottom: int,
    depth_array1_left: int,
    depth_array1_right: int,
    depth_array2_top: int,
    depth_array2_bottom: int,
    depth_array2_left: int,
    depth_array2_right: int,
    depth_array3_top: int,
    depth_array3_bottom: int,
    depth_array3_left: int,
    depth_array3_right: int,
    depth_array4_top: int,
    depth_array4_bottom: int,
    depth_array4_left: int,
    depth_array4_right: int,
):
    """Crops four depth arrays according to a shared sensor configuration.

    This function applies consistent cropping to each input array using parameters from the provided sensor configuration object.

    Args:
        depth_array1: First depth array to crop.
        depth_array2: Second depth array to crop.
        depth_array3: Third depth array to crop.
        depth_array4: Fourth depth array to crop.
        SENSOR_CONFIGURATION: Object containing cropping parameters for each sensor.

    Returns:
        Tuple of four cropped depth arrays.
    """
    depth_array1, depth_array2, depth_array3, depth_array4 = (
        crop_depth_array_consistent(
            depth_array1,
            depth_array1_top,
            depth_array1_bottom,
            depth_array1_left,
            depth_array1_right,
            depth_array2_top,
            depth_array2_bottom,
            depth_array2_left,
            depth_array2_right,
            depth_array3_top,
            depth_array3_bottom,
            depth_array3_left,
            depth_array3_right,
            depth_array4_top,
            depth_array4_bottom,
            depth_array4_left,
            depth_array4_right,
            0,
        ),
        crop_depth_array_consistent(
            depth_array2,
            depth_array1_top,
            depth_array1_bottom,
            depth_array1_left,
            depth_array1_right,
            depth_array2_top,
            depth_array2_bottom,
            depth_array2_left,
            depth_array2_right,
            depth_array3_top,
            depth_array3_bottom,
            depth_array3_left,
            depth_array3_right,
            depth_array4_top,
            depth_array4_bottom,
            depth_array4_left,
            depth_array4_right,
            1,
        ),
        crop_depth_array_consistent(
            depth_array3,
            depth_array1_top,
            depth_array1_bottom,
            depth_array1_left,
            depth_array1_right,
            depth_array2_top,
            depth_array2_bottom,
            depth_array2_left,
            depth_array2_right,
            depth_array3_top,
            depth_array3_bottom,
            depth_array3_left,
            depth_array3_right,
            depth_array4_top,
            depth_array4_bottom,
            depth_array4_left,
            depth_array4_right,
            2,
        ),
        crop_depth_array_consistent(
            depth_array4,
            depth_array1_top,
            depth_array1_bottom,
            depth_array1_left,
            depth_array1_right,
            depth_array2_top,
            depth_array2_bottom,
            depth_array2_left,
            depth_array2_right,
            depth_array3_top,
            depth_array3_bottom,
            depth_array3_left,
            depth_array3_right,
            depth_array4_top,
            depth_array4_bottom,
            depth_array4_left,
            depth_array4_right,
            3,
        ),
    )

    return depth_array1, depth_array2, depth_array3, depth_array4


def rotate_arrays(
    depth_array1: np.ndarray,
    depth_array2: np.ndarray,
    depth_array3: np.ndarray,
    depth_array4: np.ndarray,
    angle_depth_array1: int,
    angle_depth_array2: int,
    angle_depth_array3: int,
    angle_depth_array4: int,
):
    """Rotates each of four depth arrays by a specified angle.

    This function applies a rotation to each input array if its corresponding angle is nonzero, returning the rotated arrays.

    Args:
        depth_array1: First depth array to rotate.
        depth_array2: Second depth array to rotate.
        depth_array3: Third depth array to rotate.
        depth_array4: Fourth depth array to rotate.
        angle_depth_array1: Rotation angle for the first array (degrees).
        angle_depth_array2: Rotation angle for the second array (degrees).
        angle_depth_array3: Rotation angle for the third array (degrees).
        angle_depth_array4: Rotation angle for the fourth array (degrees).

    Returns:
        Tuple of four depth arrays, each rotated by its specified angle.
    """
    if angle_depth_array1 != 0:
        depth_array1 = rotate(
            depth_array1, angle_depth_array1, reshape=False, order=1, mode="nearest"
        )
    if angle_depth_array2 != 0:
        depth_array2 = rotate(
            depth_array2, angle_depth_array2, reshape=False, order=1, mode="nearest"
        )
    if angle_depth_array3 != 0:
        depth_array3 = rotate(
            depth_array3, angle_depth_array3, reshape=False, order=1, mode="nearest"
        )
    if angle_depth_array4 != 0:
        depth_array4 = rotate(
            depth_array4, angle_depth_array4, reshape=False, order=1, mode="nearest"
        )

    return depth_array1, depth_array2, depth_array3, depth_array4


def map_invalid_to_midpoint(midpoint: int, depth_array: np.ndarray):
    """Replaces invalid (zero) values in a depth array with a specified midpoint value.

    This function ensures that all zero values in the input array are set to the given midpoint.

    Args:
        midpoint: Value to assign to invalid (zero) entries.
        depth_array: NumPy array of depth values.

    Returns:
        NumPy array with zeros replaced by the midpoint value.
    """
    depth_array[depth_array == 0] = midpoint

    return depth_array


def clip_values(depth_array: np.ndarray, min_depth_value: int, max_depth_value: int):
    """Clips the values in a depth array to a specified minimum and maximum.

    This function limits all values in the input array to be within the given range.

    Args:
        depth_array: NumPy array of depth values to clip.
        min_depth_value: Minimum allowed value.
        max_depth_value: Maximum allowed value.

    Returns:
        NumPy array with values clipped to the specified range.
    """
    return np.clip(depth_array, min_depth_value, max_depth_value)


def normalize_values_into_range(
    depth_array: np.ndarray, min_depth_value: int, max_depth_value: int
):
    """Normalizes the values in a depth array to a specified range.

    This function rescales the input array so its minimum and maximum values map to the given range, and returns the result as a uint16 array.

    Args:
        depth_array: NumPy array of depth values to normalize.
        min_depth_value: Minimum value of the target range.
        max_depth_value: Maximum value of the target range.

    Returns:
        NumPy array of type uint16 with values normalized to the specified range.
    """
    depth_min, depth_max = (
        np.min(depth_array),
        np.max(depth_array),
    )
    depth_array = (depth_array - depth_min) / (depth_max - depth_min) * (
        max_depth_value - min_depth_value
    ) + min_depth_value

    return depth_array.astype(np.uint16)


def pad_to_shape(array: np.ndarray, target_shape: tuple[int, int]):
    padded = np.zeros(target_shape, dtype=array.dtype)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


def unpad_to_shape(array: np.ndarray, target_shape: tuple[int, int]):
    return array[: target_shape[0], : target_shape[1]]


def process_from_transform_matrix(
    depth_array1: np.ndarray,
    depth_array2: np.ndarray,
    depth_array3: np.ndarray,
    depth_array4: np.ndarray,
):
    # Dynamically import or reload the modules
    matrix1 = importlib.import_module("transform_matrix_from_diff_depth_array1")
    matrix2 = importlib.import_module("transform_matrix_from_diff_depth_array2")
    matrix3 = importlib.import_module("transform_matrix_from_diff_depth_array3")
    matrix4 = importlib.import_module("transform_matrix_from_diff_depth_array4")

    importlib.reload(matrix1)
    importlib.reload(matrix2)
    importlib.reload(matrix3)
    importlib.reload(matrix4)

    # Apply transformations
    depth_array1 = depth_array1 + matrix1.transform_matrix
    depth_array1 = unpad_to_shape(depth_array1, matrix1.shape)

    depth_array2 = depth_array2 + matrix2.transform_matrix
    depth_array2 = unpad_to_shape(depth_array2, matrix2.shape)

    depth_array3 = depth_array3 + matrix3.transform_matrix
    depth_array3 = unpad_to_shape(depth_array3, matrix3.shape)

    depth_array4 = depth_array4 + matrix4.transform_matrix
    depth_array4 = unpad_to_shape(depth_array4, matrix4.shape)

    return depth_array1, depth_array2, depth_array3, depth_array4


def export_transform_matrix_to_python(
    matrix: np.ndarray, filename: str, original_shape: tuple[int, int] | None = None
):
    with open(filename, "w") as f:
        f.write("import numpy as np\n\n")
        if original_shape:
            f.write(f"shape = {original_shape}\n\n")
        f.write("transform_matrix = np.array([\n")
        for row in matrix:
            row_str = ", ".join(f"{val:5d}" for val in row)
            f.write(f"    [{row_str}],\n")
        f.write("], dtype=np.int16)\n")

    print(f"Transformation matrix exported to: {filename}")
