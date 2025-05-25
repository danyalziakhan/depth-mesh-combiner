from __future__ import annotations
import cv2

import numpy as np


def stretch_top_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Stretches the top portion of the image *outward*, keeping the bottom intact.
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
    warped = cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def stretch_bottom_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Stretches the bottom portion of the image *outward*, keeping the top intact.
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
    warped = cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def stretch_left_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Stretches the left portion of the image *outward*, keeping the right intact.
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
    warped = cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def stretch_right_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Stretches the right portion of the image *outward*, keeping the left intact.
    """

    height, width = depth_array.shape
    x_indices = np.linspace(0, 1, width)

    x_mapped = np.copy(x_indices)
    right_mask = x_indices >= split_ratio
    right_x = (x_indices[right_mask] - split_ratio) / (1 - split_ratio)
    stretched = right_x ** (1 / strength)
    x_mapped[right_mask] = split_ratio + stretched * (1 - split_ratio)

    map_x, map_y = np.meshgrid(
        x_mapped * (width - 1), np.arange(height, dtype=np.float32)
    )
    warped = cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def compress_top_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Compresses (pulls inward) the top part of the image vertically toward the center,
    keeping the bottom part unchanged.

    :param strength: float > 1, higher means stronger inward pull
    :param split_ratio: where to split top vs bottom (0 to 1)
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

    # Apply remap
    warped = cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def compress_bottom_only(depth_array, strength=2.0, split_ratio=0.5):
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
    warped = cv2.remap(
        depth_array,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def compress_left_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Compresses the left side of the image horizontally *inward toward center*,
    keeping the right side unchanged.

    :param depth_array: 2D numpy array (e.g., Kinect depth frame)
    :param strength: >1 controls compression strength
    :param split_ratio: x-axis split (0 to 1), left side ends at this fraction
    :return: warped depth array
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

    # Apply remap
    warped = cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def compress_right_only(depth_array, strength=2.0, split_ratio=0.5):
    """
    Compresses (pulls inward) the right part of the image horizontally toward the center,
    while keeping the left part unchanged.

    :param strength: float > 1, higher means stronger inward pull
    :param split_ratio: float in (0,1), where to split left vs right
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

    # Apply remap
    warped = cv2.remap(
        depth_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def crop_depth_array_consistent(
    depth_array: np.array,  # type: ignore
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
