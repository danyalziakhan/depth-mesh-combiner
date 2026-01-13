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
    crop_depth_arrays_consistent,
    map_invalid_to_midpoint,
    rotate_arrays,
    get_combined_array,
    clip_values,
    export_transform_matrix_to_python,
)
from sensor_configuration import SensorConfiguration

# ! NOT USED: Dimensions of each frame for Kinect V2.
FRAME_WIDTH = 512
FRAME_HEIGHT = 424

# ! NOT USED: Kinect V2 depth range in millimeters.
MIN_DEPTH = 500
MAX_DEPTH = 4500

MOCK_DATA_SENSOR_COUNT = 4

ERROR_LOG_FILE_PATH = Path("./error_log.txt")
DEPTH_DATA_DIR = Path("./depth_data")

# Get the current process for monitoring resource usage
PROCESS = psutil.Process(os.getpid())

FRAME_NO = 1


def show_adjustment_sliders(
    depth_arrays_without_transformation: list[np.ndarray],
    sensor_configuration: SensorConfiguration,
):
    # Keep the default depth arrays intact
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

    # Just to make sure transformation are applied correctly initially
    current_arrays[0], current_arrays[1], current_arrays[2], current_arrays[3] = (
        apply_transformations_to_depth_arrays(
            current_arrays[0],
            current_arrays[1],
            current_arrays[2],
            current_arrays[3],
            sensor_configuration,
        )
    )

    combined_initial = get_combined_array(current_arrays)

    combined_initial = map_invalid_to_midpoint(
        sensor_configuration.get_midpoint(), combined_initial
    )

    combined_initial = clip_values(
        combined_initial,
        sensor_configuration.MIN_DEPTH_VALUE,
        sensor_configuration.MAX_DEPTH_VALUE,
    )

    combined_initial = combined_initial[
        sensor_configuration.TOP_MARGIN : -sensor_configuration.BOTTOM_MARGIN,
        sensor_configuration.LEFT_MARGIN : -sensor_configuration.RIGHT_MARGIN,
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
        ("SENSOR1_TOP", sensor_configuration.SENSOR1_TOP, -424, 424),
        ("SENSOR1_BOTTOM", sensor_configuration.SENSOR1_BOTTOM, -424, 424),
        ("SENSOR1_LEFT", sensor_configuration.SENSOR1_LEFT, -512, 512),
        ("SENSOR1_RIGHT", sensor_configuration.SENSOR1_RIGHT, -512, 512),
        ("SENSOR2_TOP", sensor_configuration.SENSOR2_TOP, -424, 424),
        ("SENSOR2_BOTTOM", sensor_configuration.SENSOR2_BOTTOM, -424, 424),
        ("SENSOR2_LEFT", sensor_configuration.SENSOR2_LEFT, -512, 512),
        ("SENSOR2_RIGHT", sensor_configuration.SENSOR2_RIGHT, -512, 512),
        ("SENSOR3_TOP", sensor_configuration.SENSOR3_TOP, -424, 424),
        ("SENSOR3_BOTTOM", sensor_configuration.SENSOR3_BOTTOM, -424, 424),
        ("SENSOR3_LEFT", sensor_configuration.SENSOR3_LEFT, -512, 512),
        ("SENSOR3_RIGHT", sensor_configuration.SENSOR3_RIGHT, -512, 512),
        ("SENSOR4_TOP", sensor_configuration.SENSOR4_TOP, -424, 424),
        ("SENSOR4_BOTTOM", sensor_configuration.SENSOR4_BOTTOM, -424, 424),
        ("SENSOR4_LEFT", sensor_configuration.SENSOR4_LEFT, -512, 512),
        ("SENSOR4_RIGHT", sensor_configuration.SENSOR4_RIGHT, -512, 512),
        ("SENSOR1_ANGLE", sensor_configuration.SENSOR1_ANGLE, -10, 10),
        ("SENSOR2_ANGLE", sensor_configuration.SENSOR2_ANGLE, -10, 10),
        ("SENSOR3_ANGLE", sensor_configuration.SENSOR3_ANGLE, -10, 10),
        ("SENSOR4_ANGLE", sensor_configuration.SENSOR4_ANGLE, -10, 10),
        ("LEFT_MARGIN", sensor_configuration.LEFT_MARGIN, 0, 200),
        ("RIGHT_MARGIN", sensor_configuration.RIGHT_MARGIN, 0, 200),
        ("TOP_MARGIN", sensor_configuration.TOP_MARGIN, 0, 200),
        ("BOTTOM_MARGIN", sensor_configuration.BOTTOM_MARGIN, 0, 200),
    ]
    default_values = [v for _, v, _, _ in slider_labels]

    def apply_slider_values(val=None):
        s = [int(slider.val) for slider in sliders]

        sensor_configuration[:20] = s[:20]

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
            sensor_configuration,
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
            sensor_configuration.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            sensor_configuration.MIN_DEPTH_VALUE,
            sensor_configuration.MAX_DEPTH_VALUE,
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
            sensor_configuration.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            sensor_configuration.MIN_DEPTH_VALUE,
            sensor_configuration.MAX_DEPTH_VALUE,
        )

        combined = combined[
            sensor_configuration.TOP_MARGIN : -sensor_configuration.BOTTOM_MARGIN,
            sensor_configuration.LEFT_MARGIN : -sensor_configuration.RIGHT_MARGIN,
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
            sensor_configuration.get_midpoint(), combined
        )

        combined = clip_values(
            combined,
            sensor_configuration.MIN_DEPTH_VALUE,
            sensor_configuration.MAX_DEPTH_VALUE,
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

        print(f"New Sensor Configuration exported to: {filename}")

        for i in range(MOCK_DATA_SENSOR_COUNT):
            adjusted_depth_array = current_arrays[i]
            original_depth_array = depth_arrays_without_transformation[i]

            # Compute transform matrix
            transform_matrix = pad_to_shape(
                adjusted_depth_array,
                (
                    original_depth_array.shape[0],
                    original_depth_array.shape[1],
                ),
            ).astype(np.int32) - original_depth_array.astype(np.int32)

            # Reduce memory
            transform_matrix = transform_matrix.astype(np.int16)

            export_transform_matrix_to_python(
                transform_matrix,
                f"transform_matrix_from_diff_depth_array{i + 1}.py",
                adjusted_depth_array.shape,
            )

    ax_save = fig_sliders.add_axes((0.80, 0.01, button_width, button_height))
    btn_save = Button(
        ax_save, "Save Transformation Matrix", color="lightgray", hovercolor="0.8"
    )
    btn_save.on_clicked(save_slider_values)

    apply_slider_values()
    plt.show()


def apply_transformations_to_depth_arrays(
    depth_array1: np.ndarray,
    depth_array2: np.ndarray,
    depth_array3: np.ndarray,
    depth_array4: np.ndarray,
    sensor_configuration: SensorConfiguration,
):
    depth_array1, depth_array2, depth_array3, depth_array4 = rotate_arrays(
        depth_array1,
        depth_array2,
        depth_array3,
        depth_array4,
        sensor_configuration.SENSOR1_ANGLE,
        sensor_configuration.SENSOR2_ANGLE,
        sensor_configuration.SENSOR3_ANGLE,
        sensor_configuration.SENSOR4_ANGLE,
    )

    depth_array1, depth_array2, depth_array3, depth_array4 = (
        crop_depth_arrays_consistent(
            depth_array1,
            depth_array2,
            depth_array3,
            depth_array4,
            sensor_configuration.SENSOR1_TOP,
            sensor_configuration.SENSOR1_BOTTOM,
            sensor_configuration.SENSOR1_LEFT,
            sensor_configuration.SENSOR1_RIGHT,
            sensor_configuration.SENSOR2_TOP,
            sensor_configuration.SENSOR2_BOTTOM,
            sensor_configuration.SENSOR2_LEFT,
            sensor_configuration.SENSOR2_RIGHT,
            sensor_configuration.SENSOR3_TOP,
            sensor_configuration.SENSOR3_BOTTOM,
            sensor_configuration.SENSOR3_LEFT,
            sensor_configuration.SENSOR3_RIGHT,
            sensor_configuration.SENSOR4_TOP,
            sensor_configuration.SENSOR4_BOTTOM,
            sensor_configuration.SENSOR4_LEFT,
            sensor_configuration.SENSOR4_RIGHT,
        )
    )

    return depth_array1, depth_array2, depth_array3, depth_array4


if __name__ == "__main__":
    sensor_configuration = SensorConfiguration()
    depth_arrays_without_transformation: list[np.ndarray] = []

    try:
        with open(ERROR_LOG_FILE_PATH, "w") as file:
            file.write("")

        last_time = time.perf_counter()

        if MOCK_DATA_SENSOR_COUNT != 4:
            raise ValueError(
                "Only 4 sensors data combination is supported at the moment."
            )

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
            depth_arrays_without_transformation.append(np.array(file_data_list))

        show_adjustment_sliders(
            depth_arrays_without_transformation, sensor_configuration
        )

    except Exception:
        with open(ERROR_LOG_FILE_PATH, "a+") as file:
            print(traceback.format_exc())
            file.write(traceback.format_exc())
