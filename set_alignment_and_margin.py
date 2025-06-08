from __future__ import annotations

from pathlib import Path

import traceback

from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap

import numpy as np

import os
import time
import psutil
import ujson
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from cv_ops import (
    pad_to_shape,
    process_from_transform_matrix,
    shrink_left_only,
    shrink_top_only,
    crop_depth_arrays_consistent,
    map_invalid_to_midpoint,
    rotate_arrays,
    stretch_top_only,
    get_combined_array,
    clip_values,
    shrink_bottom_only,
)
from sensor_configuration import SensorConfiguration

# ! NOT USED: Dimensions of each frame for Kinect V2.
FRAME_WIDTH = 512
FRAME_HEIGHT = 424

# ! NOT USED: Kinect V2 depth range in millimeters.
MIN_DEPTH = 500
MAX_DEPTH = 4500

MOCK_DATA_SENSOR_COUNT = 4

SENSOR_CONFIGURATION = SensorConfiguration()

ERROR_LOG_FILE_PATH = Path("./error_log.txt")
DEPTH_DATA_DIR = Path("./depth_data")

# Get the current process for monitoring resource usage
PROCESS = psutil.Process(os.getpid())

FRAME_NO = 1

depth_arrays_without_transformation = []


# Some hardcoded adjustments for heights and tilts
def adjust_heights(depth_array1, depth_array2, depth_array3, depth_array4):
    # ******************************* SENSOR 1
    depth_array1 = depth_array1 - 25
    # * *****************************

    # * This corrects changes due to tilt
    shift_per_pixels = -5  # adds shift in millimeters
    shift_for_area = 5  # apply shift at steps of these pixels

    y = np.arange(depth_array1.shape[0])
    shift_values = (y // shift_for_area) * shift_per_pixels
    shift_matrix = np.tile(shift_values[:, np.newaxis], (1, depth_array1.shape[1]))
    depth_array1 = depth_array1 + shift_matrix

    # * gradient shift
    # * This corrects changes due to sensor not perfect straight line
    initial_shift = -45  # starting shift
    shift_per_pixels = 4  # amount to add at each step
    shift_for_area = 20  # pixels per step

    x = np.arange(depth_array1.shape[1])
    steps = x // shift_for_area
    shift_values = initial_shift + steps * shift_per_pixels
    shift_values = shift_values[::-1]  # reverse to apply from left to right
    shift_matrix = np.tile(shift_values, (depth_array1.shape[0], 1))
    depth_array1 = depth_array1 + shift_matrix

    # ******************************* SENSOR 2
    depth_array2 = depth_array2 - 40
    # *******************************

    # * gradient shift
    # * This corrects changes due to sensor not perfect straight line
    initial_shift = -145  # starting shift
    shift_per_pixels = 6  # amount to add at each step
    shift_for_area = 10  # pixels per step

    x = np.arange(depth_array2.shape[1])
    steps = x // shift_for_area
    shift_values = initial_shift + steps * shift_per_pixels
    shift_matrix = np.tile(shift_values, (depth_array2.shape[0], 1))
    depth_array2 = depth_array2 + shift_matrix

    # * gradient shift
    # * This corrects changes due to sensor not perfect straight line
    initial_shift = -20  # starting shift at top
    shift_per_pixels = 4  # amount to add at each step
    shift_for_area = 16  # pixels per step

    y = np.arange(depth_array2.shape[0])  # y-axis (rows)
    steps = y // shift_for_area
    shift_values = initial_shift + steps * shift_per_pixels
    shift_matrix = np.tile(shift_values[:, np.newaxis], (1, depth_array2.shape[1]))
    depth_array2 = depth_array2 + shift_matrix

    # ******************************* SENSOR 3
    depth_array3 = depth_array3 - 66
    # *******************************

    # * gradient shift
    # * This corrects changes due to sensor not perfect straight line
    initial_shift = -169  # starting shift at top
    shift_per_pixels = 15  # amount to add at each step
    shift_for_area = 10  # pixels per step

    y = np.arange(depth_array3.shape[0])  # y-axis (rows)
    steps = y // shift_for_area
    shift_values = initial_shift + steps * shift_per_pixels
    shift_matrix = np.tile(shift_values[:, np.newaxis], (1, depth_array3.shape[1]))
    depth_array3 = depth_array3 + shift_matrix

    # * This corrects changes due to sensor not perfect straight line
    # shift_per_pixels = -6  # adds shift in millimeters
    # shift_for_area = 120  # apply shift at steps of these pixels

    # x = np.arange(depth_array3.shape[1])
    # shift_values = (x // shift_for_area) * shift_per_pixels
    # shift_matrix = np.tile(shift_values, (depth_array3.shape[0], 1))
    # depth_array3 = depth_array3 + shift_matrix

    # # * This corrects changes due to tilt
    # initial_shift = -35  # starting shift at top
    # shift_per_pixels = 5  # amount to add at each step
    # shift_for_area = 30  # pixels per step

    # y = np.arange(depth_array3.shape[0])  # y-axis (rows)
    # steps = y // shift_for_area
    # shift_values = initial_shift + steps * shift_per_pixels
    # shift_matrix = np.tile(shift_values[:, np.newaxis], (1, depth_array3.shape[1]))
    # depth_array3 = depth_array3 + shift_matrix

    # depth_array3[:, :97] -= 65
    # depth_array3[:, 97:105] -= 80
    # depth_array3[:, 105:120] -= 60
    # depth_array3[:, :125] -= np.clip(depth_array3[:, :125] - 30, 0, None)

    # * gradient shift
    # * This corrects changes due to sensor not perfect straight line
    initial_shift = -78  # starting shift
    shift_per_pixels = 5  # amount to add at each step
    shift_for_area = 10  # pixels per step

    x = np.arange(depth_array3[:, 50:190].shape[1])
    steps = x // shift_for_area
    shift_values = initial_shift + steps * shift_per_pixels
    shift_matrix = np.tile(shift_values, (depth_array3[:, 50:190].shape[0], 1))
    depth_array3[:, 50:190] = depth_array3[:, 50:190] + shift_matrix

    # ******************************* SENSOR 4
    depth_array4 = depth_array4 - 42
    # *******************************

    # * This corrects changes due to sensor not perfect straight line
    shift_per_pixels = -16  # adds shift in millimeters
    shift_for_area = 24  # apply shift at steps of these pixels

    x = np.arange(depth_array4.shape[1])
    shift_values = (x // shift_for_area) * shift_per_pixels
    shift_matrix = np.tile(shift_values, (depth_array4.shape[0], 1))
    depth_array4 = depth_array4 + shift_matrix

    return depth_array1, depth_array2, depth_array3, depth_array4


# Some hardcoded adjustments for margins and heights
def adjust_margins(depth_array1, depth_array2, depth_array3, depth_array4):
    depth_array1, depth_array2, depth_array3, depth_array4 = (
        depth_array1.astype(float),
        depth_array2.astype(float),
        depth_array3.astype(float),
        depth_array4.astype(float),
    )

    ##### * 1
    depth_array1 = shrink_top_only(depth_array1, strength=1.50, split_ratio=0.65)

    ##### * 2
    depth_array2 = shrink_top_only(depth_array2, strength=1.18, split_ratio=0.55)

    ##### * 3
    depth_array3 = shrink_bottom_only(depth_array3, strength=1.43, split_ratio=0.38)
    depth_array3 = stretch_top_only(depth_array3, strength=1.20, split_ratio=0.30)

    ##### * 4
    depth_array4 = shrink_bottom_only(depth_array4, strength=1.13, split_ratio=0.25)
    depth_array4 = shrink_left_only(depth_array4, strength=1.13, split_ratio=0.25)
    depth_array4 = stretch_top_only(depth_array4, strength=1.80, split_ratio=0.60)
    depth_array4 = shrink_left_only(depth_array4, strength=1.16, split_ratio=0.3)

    depth_array1, depth_array2, depth_array3, depth_array4 = (
        depth_array1.astype(int),
        depth_array2.astype(int),
        depth_array3.astype(int),
        depth_array4.astype(int),
    )

    return depth_array1, depth_array2, depth_array3, depth_array4


def export_transform_matrix_to_python(matrix, filename, original_shape):
    with open(filename, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"shape = {original_shape}\n\n")
        f.write("transform_matrix = np.array([\n")
        for row in matrix:
            row_str = ", ".join(f"{val:5d}" for val in row)
            f.write(f"    [{row_str}],\n")
        f.write("], dtype=np.int16)\n")
    print(f"✅ Transformation matrix exported to: {filename}")


def show_adjustment_sliders(
    depth_arrays_without_transformation,
    SENSOR_CONFIGURATION,
):
    current_arrays = [arr.copy() for arr in depth_arrays_without_transformation]

    plt.rcParams["toolbar"] = "none"  # <- Disables toolbar
    fig_plot, ax_plot = plt.subplots(figsize=(16, 9))

    manager = plt.get_current_fig_manager()
    if not manager:
        raise ValueError("Figure manager is not available.")

    if hasattr(manager, "full_screen_toggle"):
        manager.full_screen_toggle()

    manager.set_window_title("Kinect V2 Depth Viewer (4-Stream Grid)")

    # Remove all axes decorations and have the axes fill the entire figure:
    ax_plot.set_position((0, 0, 1, 1))
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    ax_plot.set_axis_off()  # hides axes frame and ticks
    ax_plot.set_aspect("auto")  # allow image to stretch

    current_arrays[0], current_arrays[1], current_arrays[2], current_arrays[3] = (
        apply_transformations_to_depth_arrays(
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
            SENSOR_CONFIGURATION,
        )
    )

    combined_initial = get_combined_array(current_arrays)

    combined_initial = map_invalid_to_midpoint(
        SENSOR_CONFIGURATION.get_midpoint(), combined_initial
    )

    combined_initial = clip_values(
        combined_initial,
        SENSOR_CONFIGURATION.MIN_DEPTH_VALUE,
        SENSOR_CONFIGURATION.MAX_DEPTH_VALUE,
    )

    combined_initial = combined_initial[
        SENSOR_CONFIGURATION.TOP_MARGIN : -SENSOR_CONFIGURATION.BOTTOM_MARGIN,
        SENSOR_CONFIGURATION.LEFT_MARGIN : -SENSOR_CONFIGURATION.RIGHT_MARGIN,
    ]

    # Load inferno and darken it
    inferno = get_cmap("inferno")
    colors = inferno(np.linspace(0, 1, 256))
    colors[:, :3] *= 0.85  # Scale RGB values (0.0 to 1.0) to darken

    # colors = ["#5D171F", "#C2B280", "#AACB9A", "#3E8090"]
    dark_inferno = ListedColormap(colors)

    im = ax_plot.imshow(
        combined_initial,
        interpolation="nearest",
        cmap=dark_inferno,
        origin="upper",
        extent=(0, 1920, 1080, 0),
    )

    slider_labels = [
        ("SENSOR1_TOP", SENSOR_CONFIGURATION.SENSOR1_TOP, -424, 424),
        ("SENSOR1_BOTTOM", SENSOR_CONFIGURATION.SENSOR1_BOTTOM, -424, 424),
        ("SENSOR1_LEFT", SENSOR_CONFIGURATION.SENSOR1_LEFT, -512, 512),
        ("SENSOR1_RIGHT", SENSOR_CONFIGURATION.SENSOR1_RIGHT, -512, 512),
        ("SENSOR2_TOP", SENSOR_CONFIGURATION.SENSOR2_TOP, -424, 424),
        ("SENSOR2_BOTTOM", SENSOR_CONFIGURATION.SENSOR2_BOTTOM, -424, 424),
        ("SENSOR2_LEFT", SENSOR_CONFIGURATION.SENSOR2_LEFT, -512, 512),
        ("SENSOR2_RIGHT", SENSOR_CONFIGURATION.SENSOR2_RIGHT, -512, 512),
        ("SENSOR3_TOP", SENSOR_CONFIGURATION.SENSOR3_TOP, -424, 424),
        ("SENSOR3_BOTTOM", SENSOR_CONFIGURATION.SENSOR3_BOTTOM, -424, 424),
        ("SENSOR3_LEFT", SENSOR_CONFIGURATION.SENSOR3_LEFT, -512, 512),
        ("SENSOR3_RIGHT", SENSOR_CONFIGURATION.SENSOR3_RIGHT, -512, 512),
        ("SENSOR4_TOP", SENSOR_CONFIGURATION.SENSOR4_TOP, -424, 424),
        ("SENSOR4_BOTTOM", SENSOR_CONFIGURATION.SENSOR4_BOTTOM, -424, 424),
        ("SENSOR4_LEFT", SENSOR_CONFIGURATION.SENSOR4_LEFT, -512, 512),
        ("SENSOR4_RIGHT", SENSOR_CONFIGURATION.SENSOR4_RIGHT, -512, 512),
        ("SENSOR1_ANGLE", SENSOR_CONFIGURATION.SENSOR1_ANGLE, -10, 10),
        ("SENSOR2_ANGLE", SENSOR_CONFIGURATION.SENSOR2_ANGLE, -10, 10),
        ("SENSOR3_ANGLE", SENSOR_CONFIGURATION.SENSOR3_ANGLE, -10, 10),
        ("SENSOR4_ANGLE", SENSOR_CONFIGURATION.SENSOR4_ANGLE, -10, 10),
        ("LEFT_MARGIN", SENSOR_CONFIGURATION.LEFT_MARGIN, 0, 200),
        ("RIGHT_MARGIN", SENSOR_CONFIGURATION.RIGHT_MARGIN, 0, 200),
        ("TOP_MARGIN", SENSOR_CONFIGURATION.TOP_MARGIN, 0, 200),
        ("BOTTOM_MARGIN", SENSOR_CONFIGURATION.BOTTOM_MARGIN, 0, 200),
    ]
    default_values = [v for _, v, _, _ in slider_labels]

    def apply_slider_values(val=None):
        s = [int(slider.val) for slider in sliders]

        SENSOR_CONFIGURATION[:20] = s[:20]

        (
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
        ) = apply_transformations_to_depth_arrays(
            depth_arrays_without_transformation[0],
            depth_arrays_without_transformation[1],
            depth_arrays_without_transformation[2],
            depth_arrays_without_transformation[3],
            SENSOR_CONFIGURATION,
        )

        left_margin = int(sliders[20].val)
        right_margin = int(sliders[21].val)
        top_margin = int(sliders[22].val)
        bottom_margin = int(sliders[23].val)

        combined = get_combined_array(current_arrays)

        h, w = combined.shape
        if 2 * top_margin >= h or 2 * left_margin >= w:
            raise ValueError("Margins are too large for the array size.")

        combined = map_invalid_to_midpoint(
            SENSOR_CONFIGURATION.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            SENSOR_CONFIGURATION.MIN_DEPTH_VALUE,
            SENSOR_CONFIGURATION.MAX_DEPTH_VALUE,
        )

        combined = combined[
            top_margin:-bottom_margin,
            left_margin:-right_margin,
        ]

        im.set_data(combined)
        fig_plot.canvas.draw_idle()

    fig_sliders, axs = plt.subplots(
        len(slider_labels), 1, figsize=(12, len(slider_labels) * 0.22)
    )
    fig_sliders.subplots_adjust(left=0.14, right=0.95, top=0.97, bottom=0.06)
    if fig_sliders.canvas.manager is not None:
        fig_sliders.canvas.manager.set_window_title("Sensor Configuration")

    sliders = []
    for ax, (label, default_val, min_val, max_val) in zip(axs, slider_labels):
        ax.set_facecolor("lightgray")
        slider = Slider(ax, label, min_val, max_val, valinit=default_val, valstep=1)
        slider.on_changed(apply_slider_values)
        sliders.append(slider)

    button_width = 0.18
    button_height = 0.04

    load_ax = fig_sliders.add_axes((0.020, 0.01, button_width, button_height))
    load_button = Button(
        load_ax, "Load Transformation Matrix", color="lightgray", hovercolor="0.8"
    )

    def load_transformation_matrix_button_clicked(val=None):
        (
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
        ) = (
            depth_arrays_without_transformation[0],
            depth_arrays_without_transformation[1],
            depth_arrays_without_transformation[2],
            depth_arrays_without_transformation[3],
        )

        (
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
        ) = process_from_transform_matrix(
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
        )

        combined = get_combined_array(current_arrays)

        combined = map_invalid_to_midpoint(
            SENSOR_CONFIGURATION.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            SENSOR_CONFIGURATION.MIN_DEPTH_VALUE,
            SENSOR_CONFIGURATION.MAX_DEPTH_VALUE,
        )

        combined = combined[
            SENSOR_CONFIGURATION.TOP_MARGIN : -SENSOR_CONFIGURATION.BOTTOM_MARGIN,
            SENSOR_CONFIGURATION.LEFT_MARGIN : -SENSOR_CONFIGURATION.RIGHT_MARGIN,
        ]

        im.set_data(combined)
        fig_plot.canvas.draw_idle()

    load_button.on_clicked(load_transformation_matrix_button_clicked)

    load_original_ax = fig_sliders.add_axes((0.28, 0.01, button_width, button_height))
    load_original_button = Button(
        load_original_ax, "Load Original", color="lightgray", hovercolor="0.8"
    )

    def load_original_clicked(val=None):
        (
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
        ) = (
            depth_arrays_without_transformation[0],
            depth_arrays_without_transformation[1],
            depth_arrays_without_transformation[2],
            depth_arrays_without_transformation[3],
        )

        combined = get_combined_array(current_arrays)

        combined = map_invalid_to_midpoint(
            SENSOR_CONFIGURATION.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            SENSOR_CONFIGURATION.MIN_DEPTH_VALUE,
            SENSOR_CONFIGURATION.MAX_DEPTH_VALUE,
        )

        im.set_data(combined)
        fig_plot.canvas.draw_idle()

    load_original_button.on_clicked(load_original_clicked)

    reset_ax = fig_sliders.add_axes((0.54, 0.01, button_width, button_height))
    reset_button = Button(reset_ax, "Reset", color="lightgray", hovercolor="0.8")

    def reset_button_clicked(event):
        for slider, val in zip(sliders, default_values):
            slider.set_val(val)

        apply_slider_values()

    reset_button.on_clicked(reset_button_clicked)

    def save_slider_values(event):
        filename = "new_sensor_configuration.txt"
        with open(filename, "w") as f:
            for idx, (label, slider_value) in enumerate(zip(slider_labels, sliders)):
                f.write(f"{label[0]}: int = {slider_value.val}\n")

                if idx in [3, 7, 11, 15, 19]:
                    f.write("\n")

        print(f"✅ New Sensor Configuration exported to: {filename}")

        # ***************************************
        adjusted_depth_array1 = current_arrays[0]
        original_depth_array1 = depth_arrays_without_transformation[0]

        transform_matrix = pad_to_shape(
            adjusted_depth_array1,
            (
                original_depth_array1.shape[0],
                original_depth_array1.shape[1],
            ),
        ).astype(np.int32) - original_depth_array1.astype(np.int32)

        transform_matrix = transform_matrix.astype(np.int16)

        export_transform_matrix_to_python(
            transform_matrix,
            "transform_matrix_from_diff_depth_array1.py",
            adjusted_depth_array1.shape,
        )

        # ***************************************
        adjusted_depth_array2 = current_arrays[1]
        original_depth_array2 = depth_arrays_without_transformation[1]

        transform_matrix = pad_to_shape(
            adjusted_depth_array2,
            (
                original_depth_array2.shape[0],
                original_depth_array2.shape[1],
            ),
        ).astype(np.int32) - original_depth_array2.astype(np.int32)

        transform_matrix = transform_matrix.astype(np.int16)

        export_transform_matrix_to_python(
            transform_matrix,
            "transform_matrix_from_diff_depth_array2.py",
            adjusted_depth_array2.shape,
        )

        # ***************************************
        adjusted_depth_array3 = current_arrays[2]
        original_depth_array3 = depth_arrays_without_transformation[2]

        transform_matrix = pad_to_shape(
            adjusted_depth_array3,
            (
                original_depth_array3.shape[0],
                original_depth_array3.shape[1],
            ),
        ).astype(np.int32) - original_depth_array3.astype(np.int32)

        transform_matrix = transform_matrix.astype(np.int16)

        export_transform_matrix_to_python(
            transform_matrix,
            "transform_matrix_from_diff_depth_array3.py",
            adjusted_depth_array3.shape,
        )

        # ***************************************
        adjusted_depth_array4 = current_arrays[3]
        original_depth_array4 = depth_arrays_without_transformation[3]

        transform_matrix = pad_to_shape(
            adjusted_depth_array4,
            (
                original_depth_array4.shape[0],
                original_depth_array4.shape[1],
            ),
        ).astype(np.int32) - original_depth_array4.astype(np.int32)

        transform_matrix = transform_matrix.astype(np.int16)

        export_transform_matrix_to_python(
            transform_matrix,
            "transform_matrix_from_diff_depth_array4.py",
            adjusted_depth_array4.shape,
        )

    ax_save = fig_sliders.add_axes((0.80, 0.01, button_width, button_height))
    btn_save = Button(
        ax_save, "Save Transformation Matrix", color="lightgray", hovercolor="0.8"
    )
    btn_save.on_clicked(save_slider_values)

    apply_slider_values()
    plt.show()


def apply_transformations_to_depth_arrays(
    depth_array1,
    depth_array2,
    depth_array3,
    depth_array4,
    SENSOR_CONFIGURATION,
):
    depth_array1, depth_array2, depth_array3, depth_array4 = rotate_arrays(
        depth_array1,
        depth_array2,
        depth_array3,
        depth_array4,
        SENSOR_CONFIGURATION.SENSOR1_ANGLE,
        SENSOR_CONFIGURATION.SENSOR2_ANGLE,
        SENSOR_CONFIGURATION.SENSOR3_ANGLE,
        SENSOR_CONFIGURATION.SENSOR4_ANGLE,
    )

    depth_array1, depth_array2, depth_array3, depth_array4 = (
        crop_depth_arrays_consistent(
            depth_array1,
            depth_array2,
            depth_array3,
            depth_array4,
            SENSOR_CONFIGURATION,
        )
    )

    depth_array1, depth_array2, depth_array3, depth_array4 = adjust_heights(
        depth_array1, depth_array2, depth_array3, depth_array4
    )

    depth_array1, depth_array2, depth_array3, depth_array4 = adjust_margins(
        depth_array1,
        depth_array2,
        depth_array3,
        depth_array4,
    )

    return depth_array1, depth_array2, depth_array3, depth_array4


if __name__ == "__main__":
    try:
        with open(ERROR_LOG_FILE_PATH, "w") as file:
            file.write("")

        last_time = time.perf_counter()

        if MOCK_DATA_SENSOR_COUNT != 4:
            raise ValueError(
                "Only 4 sensors data combination is supported at the moment."
            )

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

        depth_arrays_without_transformation = [
            depth_array1,
            depth_array2,
            depth_array3,
            depth_array4,
        ]

        show_adjustment_sliders(
            depth_arrays_without_transformation,
            SENSOR_CONFIGURATION,
        )

    except Exception:
        with open(ERROR_LOG_FILE_PATH, "a+") as file:
            file.write(traceback.format_exc())
