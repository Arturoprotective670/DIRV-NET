"""
This script, will use the processing, transformations and data
source that defined in the config file, to generate samples of
images, and their dumped intensity data in csv format, at every
step. Also, it will apply a grid on the images before
transformation, so that the transformation effects will be more
evident.
"""

# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)
# sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root

from source.coredata.batch_transformer import BatchTransformer
from source.framework.settings.whole_config import WholeConfig
from support.scripts.tools.images_helper import *

batch_size = 5
number_of_grid_lines = 20
grid_lines_intensity = 150 / 255

config = WholeConfig()
original_data, _, _ = config.data_provider(config)
original_data = original_data[0:batch_size]
processed_data = []
grid_added_data = []

for i in range(batch_size):
    img = original_data[i, :, :]
    img = config.image_pre_processor(config)(img)
    processed_data.append(img)

    grid_spacings = (
        int(processed_data[0].shape[0] / number_of_grid_lines),
        int(processed_data[0].shape[1] / number_of_grid_lines),
    )
    img = draw_array_grid(
        img.copy(), grid_spacings, grid_intensity=grid_lines_intensity
    )
    grid_added_data.append(np.array(img))

processed_data = np.array(processed_data)
grid_added_data = np.array(grid_added_data)

batch_trans = BatchTransformer(
    control_points_grid_size=config.control_points_grid_size,
    batch_transformations_generator=lambda: config.synthetic_flow_fields(
        batch_size, processed_data[0].shape, config
    ),
)

_, transformed_data, _ = batch_trans.transform_batch(processed_data, False)
transformed_data = np.squeeze(transformed_data.numpy(), axis=-1)

# TODO: enable transformer seed reset, by making it and the config classes
_, transformed_grid_data, _ = batch_trans.transform_batch(grid_added_data, False)
transformed_grid_data = np.squeeze(transformed_grid_data.numpy(), axis=-1)

from source.framework.tools.pather import Pather
import csv
from PIL import Image, ImageDraw
from matplotlib import cm


def dump_image(
    array,
    pather: Pather,
    file_name: str = "output",
):
    with open(pather.join(file_name, "csv"), "w") as f:
        csv.writer(f, delimiter=",").writerows(array)

    array_to_binary_image(array).save(pather.join(file_name, "jpg"))


TARGET_DIR = "trans_check"
pather = Pather(None, TARGET_DIR)

# for id in range(batch_size):
#     dump_image(original_data[id], pather, f"original_{id}")
#     dump_image(processed_data[id], pather, f"processed_{id}")
#     dump_image(transformed_data[id], pather, f"transformed_{id}")
#     dump_image(grid_added_data[id], pather, f"grid_{id}")
#     dump_image(transformed_grid_data[id], pather, f"transformed_grid_{id}")

from source.framework.tools.plotting_helper import images_columns_grid_figure

images = {}

images[f"original"] = [original_data[id] for id in range(batch_size)]
images[f"processed"] = [processed_data[id] for id in range(batch_size)]
images[f"transformed"] = [transformed_data[id] for id in range(batch_size)]
images[f"grid"] = [grid_added_data[id] for id in range(batch_size)]
images[f"transformed grid"] = [transformed_grid_data[id] for id in range(batch_size)]

images_columns_grid_figure(images).savefig(pather.join("test", "jpg"))
