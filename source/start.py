"""
Purpose
-------
- A module that will start training or testing the DIRV-Net.

Contributors
------------
- TMS-Namespace
"""

import sys
from pathlib import Path

# add the parent directory to the PATH, so we can import files one level up
sys.path.append(str(Path(__file__).parents[1]))

# below is to remove the various annoying tensorflow  messages (comment them out
# if you are not sure about your environment setup yet, since it may show
# important messages for that)
import os

os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =============================================================================
from source.framework.settings.whole_config import WholeConfig
from source.framework.main.image_registration import ImageRegistration

# crete configuration object, if session id is not provided, a time stamped one
# will be generated.
# Session id is used as a directory name in the output folder, for reports.
# config = WholeConfig(session_id = "delete")
config = WholeConfig("pot_func")

# create the image registerer object, that holds all of the logic
ir = ImageRegistration(config)

# start training
ir.train()


# start testing, or load saved model to test (make sure to load the
# corresponding config file also)
# ir.test()
# ir.test(use_recurrent_refinement=True, use_per_image_recurrent_refinement=True, model_variables_directory="output/run_2024-05-23_00-36-19/saved_models/10")
