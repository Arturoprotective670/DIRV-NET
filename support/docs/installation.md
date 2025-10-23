# Installation

It is recommended to use `Miniforge` instead of `Conda` because `Conda` can be extremely slow in resolving dependencies in the current setup. See [Miniforge](https://github.com/conda-forge/miniforge).

## Compatibility

It is important to ensure that the correct versions of the packages are used. See the following links:

1. [Build from source](https://www.tensorflow.org/install/source#gpu)
2. [TensorFlow API Versions](https://www.tensorflow.org/versions)
3. [Installing TensorFlow Graphics](https://www.tensorflow.org/graphics/install)
4. [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-native)
5. [GPU, CUDA Toolkit, and CUDA Driver Requirements](https://docs.nvidia.com/deeplearning/cudnn/reference/support-matrix.html)

## Setup

### 1. Yaml Installation

It is possible to re-create the environment by importing the requirements yaml file as follows:

1. For Linux:

    ```bash
    mamba env create -f support/docs/environment-linux.yaml
    ```

2. For Windows:

    ```bash
    mamba env create -f support/docs/environment-win.yaml
    ```

### 2. Manual Setup

Note that the order of execution for the below steps is important:

1. Create and activate a new environment:

    ```bash
    mamba create -n my_env python=3.9.19
    mamba activate my_env
    ```

2. Upgrade pip:

    ```bash
    python -m pip install --upgrade pip
    ```

3. Install NVIDIA libraries:

    ```bash
    mamba install -y cudatoolkit=11.2.0
    mamba install -y cudnn=8.1.0
    mamba install -y cuda -c nvidia
    ```

4. Install TensorFlow libraries:

    ```bash
    python -m pip install "tensorflow<2.11"
    python -m pip install tensorflow-graphics==2021.12.3
    ```

5. Install additional needed packages:

    ```bash
    mamba install -y scikit-learn
    mamba install -y ipywidgets
    mamba install -y simpleitk

    python -m pip install notebook
    python -m pip install nvidia-ml-py3
    ```

    Note that the rest of the packages will be installed automatically as transitive dependencies of the above.

### 3. Verify Installation

1. Check that Nvidia drivers recognize your GPUs:

    ```bash
    nvidia-smi
    ```

2. Verify that TensorFlow detects all CPUs and GPUs:

    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
    ```

### 4. Windows OS Notes

1. New TensorFlow versions (above 2.10) require `WSL2` and/or Docker to be used on Windows with GPU. However, this does not affect this project since we are using an older version of TensorFlow.

2. On Windows, Jupyter notebooks in VS Code may not detect GPUs. To resolve this, activate your environment in the Windows command line, go to the source code directory, and start VS Code from there by running the `code .` command.

3. You may also need to manually copy the `nvml.dll` file from its installation location to the `C:\Program Files\NVIDIA Corporation` folder if TensorFlow warns you that it can't find it.

## Helpful Tools

1. When you use SSH to connect and need to execute long-running processes, you should detach your terminal from the remote session. Otherwise, if you disconnect from the remote machine, the running processes will be terminated. To avoid that, you can use tools like `Screen`. A good reference is [How To Use Linux Screen](https://linuxize.com/post/how-to-use-linux-screen/).

2. You may need to monitor your hardware usage remotely. A good tool for that is [NViTop](https://github.com/XuehaiPan/nvitop).