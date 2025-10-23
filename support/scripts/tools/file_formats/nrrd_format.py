from os.path import exists
from array import array


def read_nrrd(nrrd_file_path: str) -> array:
    if not exists(nrrd_file_path):
        raise Exception("NRRD file not found.")

    import nrrd  # docs: https://pynrrd.readthedocs.io/en/stable/reference/reading.html

    data, _ = nrrd.read(nrrd_file_path, index_order="C")

    return data
