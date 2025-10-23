from os.path import exists
import numpy as np
import h5py  # docs: https://docs.h5py.org/en/stable/quick.html#quick


def read_h5(file_path: str, key: str = None) -> np.array:
    """
    Reads a file of h5 format. It assumes it to be a 4D image (Z, Y, X, T) or a
    3D image (Z, Y, X).

    Arguments
    ---------
    - `fila_path`: the path of the file to read.
    - `key`: the key that specifies the data set to read.

    Returns
    -------
    A Numpy array of the contents in the format (Z, Y, X, T) or (Y, X, T).

    Notes
    -----
    If no key is specified, the contents of the first key will be returned.
    """

    if not exists(file_path):
        raise Exception("file not found.")

    file = h5py.File(file_path, "r")

    if key is None:
        key = list(file.keys())[0]

    data = file[key][...]  # [...] is to convert it to numpy array
    rank = len(data.shape)

    if rank == 4:
        data = data.transpose(1, 2, 0, 3)
    elif rank == 3:
        data = data.transpose(1, 2, 0)
    else:
        raise NotImplementedError

    file.close()

    return data


def write_h5(file_path, data, key: str = "ds"):
    rank = len(data.shape)

    if rank == 4:
        data = data.transpose(2, 0, 1, 3)
    elif rank == 3:
        data = data.transpose(2, 0, 1)
    else:
        raise NotImplementedError

    with h5py.File(file_path, "w") as f:
        f.create_dataset(key, data=data)
