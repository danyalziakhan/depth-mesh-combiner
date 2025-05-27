from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ujson

from cv_ops import (
    clip_values,
    get_combined_array,
    map_invalid_to_midpoint,
    process_from_transform_matrix,
)
from kinect_data import SensorsConfiguration


SENSORS_CONFIGURATION = SensorsConfiguration()

DEPTH_DATA_DIR = Path("./depth_data")

FRAME_NO = 1

if __name__ == "__main__":
    try:
        with open(
            os.path.join(
                DEPTH_DATA_DIR,
                f"depth_array_int_{FRAME_NO}_array_1.txt",
            ),
            "r",
        ) as f:
            file_data = f.read()
    except FileNotFoundError as e:
        raise ValueError(
            f"Depth data in {DEPTH_DATA_DIR} directory is not present. Run the program with actual sensors to generate depth data."
        ) from e

    file_data_list = ujson.loads(file_data)
    depth_array1 = np.array(file_data_list)

    try:
        with open(
            os.path.join(
                DEPTH_DATA_DIR,
                f"depth_array_int_{FRAME_NO}_array_2.txt",
            ),
            "r",
        ) as f:
            file_data = f.read()
    except FileNotFoundError as e:
        raise ValueError(
            f"Depth data in {DEPTH_DATA_DIR} directory is not present. Run the program with actual sensors to generate depth data."
        ) from e

    file_data_list = ujson.loads(file_data)
    depth_array2 = np.array(file_data_list)

    try:
        with open(
            os.path.join(
                DEPTH_DATA_DIR,
                f"depth_array_int_{FRAME_NO}_array_3.txt",
            ),
            "r",
        ) as f:
            file_data = f.read()
    except FileNotFoundError as e:
        raise ValueError(
            f"Depth data in {DEPTH_DATA_DIR} directory is not present. Run the program with actual sensors to generate depth data."
        ) from e

    file_data_list = ujson.loads(file_data)
    depth_array3 = np.array(file_data_list)

    try:
        with open(
            os.path.join(
                DEPTH_DATA_DIR,
                f"depth_array_int_{FRAME_NO}_array_4.txt",
            ),
            "r",
        ) as f:
            file_data = f.read()
    except FileNotFoundError as e:
        raise ValueError(
            f"Depth data in {DEPTH_DATA_DIR} directory is not present. Run the program with actual sensors to generate depth data."
        ) from e

    file_data_list = ujson.loads(file_data)
    depth_array4 = np.array(file_data_list)

    (
        depth_array1,
        depth_array2,
        depth_array3,
        depth_array4,
    ) = process_from_transform_matrix(
        depth_array1,
        depth_array2,
        depth_array3,
        depth_array4,
    )

    combined = get_combined_array(
        [
            depth_array1,
            depth_array2,
            depth_array3,
            depth_array4,
        ]
    )

    combined = map_invalid_to_midpoint(SENSORS_CONFIGURATION.get_midpoint(), combined)

    combined = clip_values(
        combined,
        SENSORS_CONFIGURATION.MIN_DEPTH_VALUE,
        SENSORS_CONFIGURATION.MAX_DEPTH_VALUE,
    )

    combined = combined[
        SENSORS_CONFIGURATION.TOP_MARGIN : -SENSORS_CONFIGURATION.BOTTOM_MARGIN,
        SENSORS_CONFIGURATION.LEFT_MARGIN : -SENSORS_CONFIGURATION.RIGHT_MARGIN,
    ]

    from transform_matrix_combined_adjusted import transform_matrix

    combined = combined + transform_matrix

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(combined, cmap="viridis")
    plt.colorbar(im)
    im.format_cursor_data = (
        lambda val: f"{int(val)}"
    )  # Disable exponent formatting # type: ignore

    plt.show()
