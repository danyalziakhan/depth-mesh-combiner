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

# Dimensions of each frame for Kinect V2.
FRAME_WIDTH = 512
FRAME_HEIGHT = 424

# Customize this to control how many neighboring pixels are affected
NEIGHBORHOOD_RADIUS = 3  # 1 = 3x3, 2 = 5x5, 0 = only center pixel
PIXELS_TO_CHANGE = 30

SENSORS_CONFIGURATION = SensorsConfiguration()

DEPTH_DATA_DIR = Path("./depth_data")

FRAME_NO = 1


def update_display():
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


def on_click(event):
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


def on_key(event):
    global NEIGHBORHOOD_RADIUS
    y, x = selected_pixel

    if event.key == "[":
        if NEIGHBORHOOD_RADIUS > 0:
            NEIGHBORHOOD_RADIUS -= 1
            print(
                f"⬅️ Radius decreased to {NEIGHBORHOOD_RADIUS} → {(2 * NEIGHBORHOOD_RADIUS + 1)}x{(2 * NEIGHBORHOOD_RADIUS + 1)}"
            )
        else:
            print("⚠️ Radius already at minimum (0)")
        update_display()
        return

    elif event.key == "]":
        NEIGHBORHOOD_RADIUS += 1
        print(
            f"➡️ Radius increased to {NEIGHBORHOOD_RADIUS} → {(2 * NEIGHBORHOOD_RADIUS + 1)}x{(2 * NEIGHBORHOOD_RADIUS + 1)}"
        )
        update_display()
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
        print("✅ Exported: transform_matrix_combined_adjusted.py")
    else:
        print("Use '+', '-', '[', ']', or 's'.")

    update_display()


def export_transform_matrix_to_python(matrix, filename):
    with open(filename, "w") as f:
        f.write("import numpy as np\n\n")
        f.write("transform_matrix = np.array([\n")
        for row in matrix:
            row_str = ", ".join(f"{val:5d}" for val in row)
            f.write(f"    [{row_str}],\n")
        f.write("], dtype=np.int16)\n")
    print(f"✅ Transformation matrix exported to: {filename}")


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

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(combined, cmap="viridis")
    plt.colorbar(im)
    im.format_cursor_data = (
        lambda val: f"{int(val)}"
    )  # Disable exponent formatting # type: ignore

    transform_matrix = np.zeros((combined.shape[0], combined.shape[1]), dtype=np.int16)

    selected_pixel = [None, None]

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_display()
    plt.show()
