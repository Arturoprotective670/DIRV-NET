"""
Purpose
-------
- Shorthand functions to generate Monotone images, histograms, heat-maps, vector
fields and line plots.
- Draw a grid over an image.
- Generate a grid of images of a unified size.
- Generate shared color bar map for a set of heat images.
- Convert Matplotlib figures to Tensorboard images.

Contributors
------------
- TMS-Namespace.
- Claudio Fanconi.
"""

from typing import Any, Callable, List, Optional, Tuple
from numpy.typing import NDArray

import io
import math
import numpy as np
import platform
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont


def grid_figure(
    grid_shape: Tuple[int, int],
    cell_drawers: Callable[[int, int], plt.Artist],
    cell_size_inches: Tuple[float, float] = (3, 3),
) -> Tuple[Figure, NDArray[Any]]:
    """
    Generates a grid of images of a unified size, as a MatplotLib Figure object.

    Parameters
    ----------
    - `grid_shape` : A tuple representing the number of rows and columns in the grid.
    - `cell_drawers` : A function that will plot the images in each cell of the grid,
    it takes two arguments (height, width), and returns a one of Matplotlib plots,
    that represents the image array that should be inserted into the cell with of the current
    width/height coordinates.
    - `cell_size_inches` : The size of grid cells in inches.

    Returns
    -------
    - A matplotlib figure of the grid.
    - An array of Matplotlib plots representing the images in the grid.
    """

    figure = plt.figure(
        figsize=(
            grid_shape[1] * cell_size_inches[0],
            grid_shape[0] * cell_size_inches[1],
        )
    )

    drawn_images = np.zeros(grid_shape, dtype=object)

    for h in range(grid_shape[0]):
        for w in range(grid_shape[1]):
            figure.add_subplot(
                grid_shape[0],
                grid_shape[1],
                w + h * grid_shape[1] + 1,
            )
            drawn_images[h, w] = cell_drawers(h, w)

    figure.tight_layout()

    return figure, drawn_images


def heat_map_drawer(
    image: NDArray[np.float32],
    title: str = "",
    display_grid: bool = False,
    display_axis: bool = False,
    title_font_size: float = 10,
    color_map_name: str = "None",
    color_map_min_val: int = 0,
    color_map_max_val: int = 255,
) -> Optional[AxesImage]:
    """
    Draw a heat map of an image using matplotlib.

    Parameters
    ----------
    - `image` : 2D NumPy array, the image to draw.
    - `title` : str, plot title.
    - `display_grid` : bool, display grid.
    - `display_axis` : bool, display axis.
    - `title_font_size` : float, font size of the plot title.
    - `color_map_name` : str, name of the color map used in the plot.
    - `color_map_min_val` : int, minimum value of the color map.
    - `color_map_max_val` : int, maximum value of the color map.

    Returns
    -------
    - A matplotlib AxesImage or None if image is None.
    """
    if image is not None:
        plt.grid(display_grid)

        if display_axis is False:
            plt.axis("off")

        color_map = plt.get_cmap(color_map_name)

        plt.title(str(title), fontsize=title_font_size)

        norm = plt.Normalize(vmin=color_map_min_val, vmax=color_map_max_val)
        return plt.imshow(image.astype(np.float32), cmap=color_map, norm=norm)
    else:
        plt.axis("off")
        return None


def annotated_heat_map_drawer(
    image: NDArray[np.float32],
    box_text: str,
    title: str = "",
    display_grid: bool = False,
    display_axis: bool = False,
    title_font_size: float = 10,
    color_map_name: str = "None",
    color_map_min_val: int = 0,
    color_map_max_val: int = 255,
    box_text_color: Tuple[int, int, int, int] = (0, 0, 0, 200),
    box_font_size: int = 12,
    box_background_color: Tuple[int, int, int, int] = (200, 200, 200, 200),
    box_margin: int = 4,
    box_padding: int = 4,
    corner_radius: int = 5,
) -> Optional[AxesImage]:
    """
    Draw a heat map of an image using matplotlib, with text annotations in the lower right corner.

    Parameters
    ----------
    - `image` : 2D NumPy array, the image to draw.
    - `box_text` : The text to be drawn.
    - `title` : str, plot title.
    - `display_grid` : bool, display grid.
    - `display_axis` : bool, display axis.
    - `title_font_size` : float, font size of the plot title.
    - `color_map_name` : str, name of the color map used in the plot.
    - `color_map_min_val` : int, minimum value of the color map.
    - `color_map_max_val` : int, maximum value of the color map.
    - `box_text_color` : The color of the text in RGBA format. Default is black with full opacity.
    - `box_text_size` : The font size of the text.
    - `box_background_color` : The background color of the text box in RGBA format.
    - `box_margin` : The margin between the text box and the edges of the image.
    - `box_padding` : The padding between the text and the edges of the text box.
    - `corner_radius` : The radius of the corners of the text box.

    Returns
    -------
    - A matplotlib AxesImage or None if image is None.
    """

    ax_image = heat_map_drawer(
        image,
        title,
        display_grid,
        display_axis,
        title_font_size,
        color_map_name,
        color_map_min_val,
        color_map_max_val,
    )

    if ax_image is None:
        return None

    image_arr = _ax_image_to_array(ax_image)

    annotated_image_arr = _draw_text_box(
        image_arr,
        box_text,
        box_text_color,
        box_font_size,
        box_background_color,
        box_margin,
        box_padding,
        corner_radius,
    )

    # convert to AxesImage
    return plt.imshow(annotated_image_arr)


def vector_field_drawer(
    X: NDArray,
    Y: NDArray,
    title: str,
    display_grid: bool,
    display_axis: bool,
    title_font_size: float,
    color_map_name: str,
    max_length: float,
) -> Optional[AxesImage]:
    """
    Draw a vector field with provided data.

    Parameters
    ----------
    - `X`: 2D NumPy array, X component of the vector field.
    - `Y`: 2D NumPy array, Y component of the vector field.
    - `title`: str, plot title.
    - `display_grid`: bool, display grid.
    - `display_axis`: bool, display axis.
    - `title_font_size`: float, font size of the plot title.
    - `color_map_name`: str, name of the color map used in the plot.
    - `max_length`: float, maximum length of the arrows in the plot.

    Returns
    -------
    - A matplotlib plot or None if X or Y is None.
    """
    # TODO: extract the max_length from the max  of vector field
    if X is not None and Y is not None:
        plt.grid(display_grid)

        if display_axis is False:
            plt.axis("off")

        plt.title(str(title), fontsize=title_font_size)

        shape = np.shape(X)
        x = np.linspace(0, shape[1], shape[1])
        # quiver uses left-bottom corner as coordinate origin, but we need upper-left one
        y = np.linspace(shape[0], 0, shape[0])

        color_map = plt.get_cmap(color_map_name)

        # max_length / 3 just to make arrows slightly longer
        scale = shape[0] * (max_length - max_length / 3.5)

        colors = np.sqrt(X**2 + Y**2)

        norm = plt.Normalize(vmin=0, vmax=max_length)

        # we invert X, due to origin position
        return plt.quiver(x, y, -X, Y, colors, cmap=color_map, scale=scale, norm=norm)
    else:
        plt.axis("off")
        return None


def histogram_drawer(
    data: NDArray,
    bins_count: int,
    title: str,
    display_grid: bool,
    title_font_size: float,
    alpha: float,
) -> Optional[AxesImage]:
    """
    Draw a histogram with provided data.

    Parameters
    ----------
    - `data`: Data to be processed.
    - `bins_count`: Number of bins in histogram.
    - `title`: Plot title.
    - `display_grid`: Display grid.
    - `title_font_size`: Font size of title.
    - `alpha`: Transparency channel for plot.

    Returns
    -------
    - A matplotlib plot, or None if data is None.
    """
    if data is not None:
        plt.grid(display_grid)

        plt.title(str(title), fontsize=title_font_size)

        hist, bin_edges = np.histogram(data, bins=bins_count)
        return plt.hist(bin_edges[:-1], bin_edges, alpha=alpha, weights=hist)
    else:
        plt.axis("off")
        return None


def line_plot_drawer(
    data_list: List[List[float]],
    title: str,
    labels_list: List[str],
    display_grid: bool,
    show_axis: bool,
    title_font_size: int,
    max_x: Optional[int] = None,
    shared_x_data : Optional[list[float]] = None
) -> Optional[AxesImage]:
    """
    Draw a line plot with provided data.

    Parameters
    ----------
    - `data_list` : A list of data to plot, list of lists of floats.
    - `title` : A string representing the title of the plot.
    - `labels_list` : A list of string representing the label of each line in
      the plot.
    - `display_grid` : A boolean indicating whether to display grid or not.
    - `show_axis` : A boolean indicating whether to show axis or not.
    - `title_font_size` : An integer representing the font size of the title.
    - `max_x` : An integer representing the maximum x value of the plot, None
      for automatic calculation.
    - `shared_x_data` : A list of floats representing the x values of the
      data in the plot, should not be used with `max_x`.

    Returns
    -------
    - A matplotlib plot, or None if data is None.

    """
    if data_list is not None:
        plt.grid(display_grid)

        if not show_axis:
            plt.axis("off")

        plt.title(str(title), fontsize=title_font_size)

        plot_index = 0
        for data in data_list:
            if labels_list is None:
                label = ""
            else:
                label = labels_list[plot_index]

            if shared_x_data is not None:
                x_data = shared_x_data
            else:
                if max_x is None:
                    data_max_x = len(data)
                else:
                    data_max_x = max_x

                x_data = list(np.arange(0, data_max_x, data_max_x / len(data)))

            # ensure that x,y has same amount of points
            if len(x_data) > len(data):
                x_data.remove(x_data[-1])
            elif len(x_data) < len(data):
                x_data.append(x_data[-1] + 1)

            plt.plot(x_data, data, label=label)
            plot_index += 1

        if labels_list is not None:
            plt.legend()

        plt.gca().xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # make axis ticks at integers only

    else:
        plt.axis("off")
        return None


def setup_grid_figure_color_bar(
    figure: plt.Figure,
    grid_columns_count: int,
    reference_image: plt.Artist,
    color_map_min_val: float,
    color_map_max_val: float,
    step: int = 1,
) -> None:
    """
    Setup a color bar on a figure, with customized ticks.

    Arguments:
    ----------
    - `figure`: the figure where color bar will be added.
    - `grid_columns_count`: the number of columns in the grid, used to adjust left position of color bar.
    - `reference_image`: the reference image for the color bar (since it can be drawn on grid of images, where some of then uses a unified bar).
    - `color_map_min_val`: the minimum value of color map.
    - `color_map_max_val`: the maximum value of color map.
    - `step`: the step of the bar ticks.

    Returns:
    --------
    - None
    """

    figure.subplots_adjust(left=0.4 / grid_columns_count)
    bar_axes = figure.add_axes([0.01, 0.15, 0.08 / grid_columns_count, 0.7])

    abs_min = np.abs(color_map_min_val)
    sign_min = int(np.sign(color_map_min_val))

    abs_max = np.abs(color_map_max_val)
    sign_max = int(np.sign(color_map_max_val))

    # enforce showing min/max values on color bar
    ticks = list(range(sign_min * int(abs_min), sign_max * int(abs_max) + 1, step))

    # if min/max value is not integer, it will be not in above, so we need to include it
    INT_THRESHOLD = 0.01
    decimals = abs_min - int(abs_min)
    if decimals >= INT_THRESHOLD:
        if (
            decimals < 0.4
        ):  # if first tick is too close to the next one, remove the next one
            ticks = ticks[1:]
        ticks = [sign_min * math.floor(abs_min * 100) / 100] + ticks

    decimals = abs_max - int(abs_max)
    if decimals >= INT_THRESHOLD:
        if decimals < 0.4:
            ticks = ticks[:-1]
        ticks = ticks + [sign_max * math.floor(abs_max * 100) / 100]

    color_bar = figure.colorbar(reference_image, cax=bar_axes, ticks=ticks)
    color_bar.ax.set_yticklabels(ticks)
    color_bar.ax.set_yticklabels(ticks)


def draw_array_grid(
    array: np.array,
    grid_spacings: Tuple[int, int] = (4, 4),
    grid_lines_intensity: Optional[int] = 10,
) -> np.array:
    """
    Draws a simple grid on the given 2D array.

    Arguments
    ---------
    - `array`: the array that will be modified.
    - `grid_spacings`: the distance between the grid lines, horizontally and
    vertically.
    - `grid_intensity`: the new value of the array in the locations of the grid,
    if `None`, the value will inverted supposing max value of `1`.

    Return
    ------
    - The modified array with the gird.

    Note
    ----
    The grid will not be generated on the borders of the array.
    """

    h, w = array.shape[0], array.shape[1]
    y, x = grid_spacings[0], grid_spacings[1]

    # TODO: use masks for this
    while y < h:
        for x_i in range(w):
            array[y, x_i] = (
                1 - array[y, x_i]
                if grid_lines_intensity is None
                else grid_lines_intensity
            )
        y += grid_spacings[0]

    while x < w:
        for y_i in range(h):
            array[y_i, x] = (
                1 - array[y_i, x]
                if grid_lines_intensity is None
                else grid_lines_intensity
            )
        x += grid_spacings[1]

    return array


def draw_batch_tensor_grid(
    data: tf.Tensor,
    grid_spacings: Tuple[int, int] = (4, 4),
    grid_lines_intensity: Optional[int] = 10,
) -> tf.Tensor:
    dtype = data.dtype
    data = data[..., 0].numpy()
    grided = []

    for i in range(data.shape[0]):
        grided.append(
            draw_array_grid(data[i, ...], grid_spacings, grid_lines_intensity)
        )

    return tf.cast(np.expand_dims(np.array(grided), axis=-1), dtype=dtype)


def figure_to_tensorboard_image(figure: plt.Figure) -> tf.Tensor:
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, axis=0)

    return image


def _draw_text_box(
    image: NDArray,
    text: str,
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 200),
    text_size: int = 12,
    box_background_color: Tuple[int, int, int, int] = (200, 200, 200, 200),
    margin: int = 4,
    padding: int = 4,
    corner_radius: int = 3,
) -> NDArray:
    """
    Draws text in a rectangular box on the lower right corner of the image with transparency support.

    Arguments
    ----------
    - `image` : The input image array in RGBA format.
    - `text` : The text to be drawn.
    - `text_color` : The color of the text in RGBA format. Default is black with full opacity.
    - `text_size` : The font size of the text.
    - `box_background_color` : The background color of the text box in RGBA format.
    - `margin` : The margin between the text box and the edges of the image.
    - `padding` : The padding between the text and the edges of the text box.
    - `corner_radius` : The radius of the corners of the text box.

    Returns
    -------
    - The output image array with the text box drawn.
    """

    if image.shape[2] != 4:
        raise ValueError("Input image must be in RGBA format")

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")

    # we need to load some font here, and the problem that this is os specific
    system = platform.system()
    if system == "Windows":
        font = ImageFont.truetype("arial.ttf", text_size)
    elif system == "Linux":
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", text_size
        )
    else:
        raise OSError(f"Do know how to load a font on {system}.")

    # Get text box size
    text_bbox = draw.textbbox((0, 0), text, font=font)

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate rectangle position and size
    rect_width = text_width + 2 * padding
    rect_height = text_height + 2 * padding

    rect_x = pil_image.width - rect_width - margin
    rect_y = pil_image.height - rect_height - margin

    # Create an overlay image for the rectangle with transparency
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))

    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    if corner_radius > 0:
        overlay_draw.rounded_rectangle(
            [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
            radius=corner_radius,
            fill=box_background_color,
        )
    else:
        overlay_draw.rectangle(
            [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
            fill=box_background_color,
        )

    # Composite the overlay with the original image
    combined_image = Image.alpha_composite(pil_image, overlay)

    # Draw text on the combined image
    draw = ImageDraw.Draw(combined_image, "RGBA")
    text_x = rect_x + (rect_width - text_width) // 2 - text_bbox[0]
    text_y = rect_y + (rect_height - text_height) // 2 - text_bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Convert the PIL image back to numpy array
    result_image = np.array(combined_image)

    return result_image


def _ax_image_to_array(ax_image: AxesImage) -> NDArray:
    """
    Converts the RGBA AxesImage object to a NumPy array.
    """
    # render image by drawing the canvas
    ax_image.figure.canvas.draw()

    # Extract the image data from the AxesImage object
    data = ax_image.make_image(renderer=ax_image.figure.canvas.get_renderer())

    # Y-axis is flipped by now
    return np.array(data[0])[::-1, ...]
