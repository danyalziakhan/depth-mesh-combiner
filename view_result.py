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
from sensor_configuration import SensorConfiguration


MOCK_DATA_SENSOR_COUNT = 4
DEPTH_DATA_DIR = Path("./depth_data")

FRAME_NO = 1

if __name__ == "__main__":
    sensor_configuration = SensorConfiguration()
    depth_arrays = []

    for i in range(1, MOCK_DATA_SENSOR_COUNT + 1):
        file_path = os.path.join(
            DEPTH_DATA_DIR,
            f"depth_array_int_{FRAME_NO}_array_{i}.txt",
        )

        try:
            with open(file_path, "r") as f:
                file_data = f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"Depth data in {DEPTH_DATA_DIR} directory is not present. "
                "Run the program with actual sensors to generate depth data."
            ) from e

        file_data_list = ujson.loads(file_data)
        depth_arrays.append(np.array(file_data_list))

    (depth_arrays[0], depth_arrays[1], depth_arrays[2], depth_arrays[3]) = (
        process_from_transform_matrix(
            depth_arrays[0],
            depth_arrays[1],
            depth_arrays[2],
            depth_arrays[3],
        )
    )

    combined = get_combined_array(depth_arrays)

    combined = map_invalid_to_midpoint(sensor_configuration.get_midpoint(), combined)

    combined = clip_values(
        combined,
        sensor_configuration.MIN_DEPTH_VALUE,
        sensor_configuration.MAX_DEPTH_VALUE,
    )

    combined = combined[
        sensor_configuration.TOP_MARGIN : -sensor_configuration.BOTTOM_MARGIN,
        sensor_configuration.LEFT_MARGIN : -sensor_configuration.RIGHT_MARGIN,
    ]

    from transform_matrix_combined_adjusted import transform_matrix

    combined = combined + transform_matrix

    fig, ax = plt.subplots(figsize=(16, 9))

    manager = plt.get_current_fig_manager()
    if not manager:
        raise ValueError("Figure manager is not available.")

    if hasattr(manager, "full_screen_toggle"):
        manager.full_screen_toggle()

    manager.set_window_title("Kinect V2 Depth Viewer (4-Stream Grid)")

    im = ax.imshow(combined, cmap="viridis")
    plt.colorbar(im)
    im.format_cursor_data = (
        lambda val: f"{int(val)}"
    )  # Disable exponent formatting # type: ignore

    plt.show()
