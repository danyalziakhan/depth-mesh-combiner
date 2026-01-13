from __future__ import annotations
from functools import partial
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
    export_transform_matrix_to_python,
)
from sensor_configuration import SensorConfiguration

# Dimensions of each frame for Kinect V2.
FRAME_WIDTH = 512
FRAME_HEIGHT = 424

# Customize this to control how many neighboring pixels are affected
NEIGHBORHOOD_RADIUS = 3  # 1 = 3x3, 2 = 5x5, 0 = only center pixel
PIXELS_TO_CHANGE = 30

MOCK_DATA_SENSOR_COUNT = 4
DEPTH_DATA_DIR = Path("./depth_data")

FRAME_NO = 1


def update_display(combined, transform_matrix):
    modified = combined + transform_matrix
    im.set_data(modified)
    ax.set_title(
        f"Click pixel. '+' or '-' modifies. '[' or ']' changes block size ({2 * NEIGHBORHOOD_RADIUS + 1}x{2 * NEIGHBORHOOD_RADIUS + 1}). 's' saves."
    )
    fig.canvas.draw()


def get_pixel_block(center_y, center_x, radius):
    ys = range(max(0, center_y - radius), min(FRAME_HEIGHT, center_y + radius + 1))
    xs = range(max(0, center_x - radius), min(FRAME_WIDTH, center_x + radius + 1))
    return [(y, x) for y in ys for x in xs]


def on_click(event, combined, transform_matrix):
    if event.inaxes != ax:
        return
    y, x = int(event.ydata), int(event.xdata)
    selected_pixel[0], selected_pixel[1] = y, x  # type: ignore
    print(f"Selected center: ({y}, {x})")
    block = get_pixel_block(y, x, NEIGHBORHOOD_RADIUS)
    for by, bx in block:
        print(
            f" └─ Will affect pixel ({by}, {bx}) | Depth: {combined[by, bx]} | Transform: {transform_matrix[by, bx]}"
        )


def on_key(event, combined, transform_matrix):
    global NEIGHBORHOOD_RADIUS
    y, x = selected_pixel

    if event.key == "[":
        if NEIGHBORHOOD_RADIUS > 0:
            NEIGHBORHOOD_RADIUS -= 1
            print(
                f"⬅Radius decreased to {NEIGHBORHOOD_RADIUS} → {(2 * NEIGHBORHOOD_RADIUS + 1)}x{(2 * NEIGHBORHOOD_RADIUS + 1)}"
            )
        else:
            print("Radius already at minimum (0)")
        update_display(combined, transform_matrix)
        return

    elif event.key == "]":
        NEIGHBORHOOD_RADIUS += 1
        print(
            f"Radius increased to {NEIGHBORHOOD_RADIUS} → {(2 * NEIGHBORHOOD_RADIUS + 1)}x{(2 * NEIGHBORHOOD_RADIUS + 1)}"
        )
        update_display(combined, transform_matrix)
        return

    if y is None or x is None:
        print("No pixel selected. Click to select one.")
        return

    block = get_pixel_block(y, x, NEIGHBORHOOD_RADIUS)

    if event.key == "+":
        for by, bx in block:
            transform_matrix[by, bx] += PIXELS_TO_CHANGE
        print(f"Added +{PIXELS_TO_CHANGE} to {len(block)} pixels")
    elif event.key == "-":
        for by, bx in block:
            transform_matrix[by, bx] -= PIXELS_TO_CHANGE
        print(f"Subtracted -{PIXELS_TO_CHANGE} from {len(block)} pixels")
    elif event.key == "s":
        export_transform_matrix_to_python(
            transform_matrix, "transform_matrix_combined_adjusted.py"
        )
        print("Exported: transform_matrix_combined_adjusted.py")
    else:
        print("Use '+', '-', '[', ']', or 's'.")

    update_display(combined, transform_matrix)


if __name__ == "__main__":
    sensor_configuration = SensorConfiguration()

    depth_arrays = []

    for i in range(1, MOCK_DATA_SENSOR_COUNT):
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

    transform_matrix = np.zeros((combined.shape[0], combined.shape[1]), dtype=np.int16)

    selected_pixel = [None, None]

    on_click = partial(on_click, combined=combined, transform_matrix=transform_matrix)
    on_key = partial(on_key, combined=combined, transform_matrix=transform_matrix)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_display(combined, transform_matrix)
    plt.show()
