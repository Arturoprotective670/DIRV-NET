# Usage

## Learning Procedure

1. Edit the framework configuration file located in `source/framework/settings/whole_config.py` according to your needs. This file defines all hyperparameters, as well as the dataset used and various reporting settings.
2. Set up the starting file located in `source/start.py` by loading the initializing configuration file, possibly with some custom session ID, and initializing the main image registration class, as well as invoking the learning method.

   ```python
   config = WholeConfig("my_run_id")
   ir = ImageRegistration(config)
   ir.train()
   ```

   If `my_run_id` is not set, it will be generated automatically based on the timestamp and whether we are running in debug mode.

3. Execute the starting script:

   ```bash
   python3 -O source/start.py
   ```

   Use `-O` to turn debug mode off, for optimization and better performance.

4. This will start the model training process, and it will create a folder `output/my_run_id` where all the training results, the configuration file `whole_config.py`, model variables, and reports are saved.

## Inference Procedure

1. For inference on a test dataset, the `ir.test()` function can be invoked in the starting file. A path to the saved model parameters of a pre-trained model can also be passed.

2. Another option is to perform inference on a single tensor of images you provide; for that, use `ir.single_infer()`.

## Reports

1. A text-based logging file (the same as the console output) can be found in `output/my_run_id/log.txt`.

2. To start TensorBoard, use the following command:

    ```bash
    tensorboard --samples_per_plugin=images=100 --logdir output/
    ```

    By using the `--samples_per_plugin=images=100` argument, you will force TensorBoard to load all the reported images in all epochs in sliders; otherwise, it will skip some automatically.

    This will port-forward from the TensorBoard server to your local device and provide you with a link to launch TensorBoard in your browser.

    TensorBoard reports are saved in `output/my_run_id/tensorboard_logs/`; however, note that TensorBoard will find all its reports in `output/` automatically and provide a way to filter them in its UI, so there is no need to specify single run.

3. Model parameters are saved in `output/my_run_id/saved_models/`.
4. The reported losses during training, validation, or testing are dumped in `.csv` format in `output/my_run_id/losses/`.
