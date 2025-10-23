from __future__ import annotations

from os import makedirs, walk
from os.path import join, exists
from pathlib import Path
from typing import List, Optional


class Pather:
    """
    Purpose
    -------
    A helper class to represent easily file paths, create directories, and
    return the contents.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self, *dirs_hierarchy: str, default_extension: Optional[str] = None
    ) -> None:
        """
        Initialize the `Pather` class.

        Arguments
        ---------
        - `default_extension`: is the default extension that will be used, if
        none is provided when join function is called.
        - `dirs_hierarchy`: arguments that defines the nested directories names
        that should all paths be created starting from it.

        Note
        ----
        If the directories hierarchy dose not exists, they will be created.
        """
        self.default_extension = default_extension

        self._path_obj = Path(join(*dirs_hierarchy))
        self.full_path = str(self._path_obj.absolute())

        if "." in self.full_path:
            self._is_file = True
            self.directory = str(self._path_obj.parent.absolute())
            self.full_name = self._path_obj.name
            self.extension = self._path_obj.suffix
            self.name = self.full_name[: -len(self.extension)]
        else:
            self._is_file = False
            self.directory = str(self._path_obj.absolute())
            self.full_name = self._path_obj.name
            self.extension = None
            self.name = self.full_name
            # self.full_path = str(self._path_obj.absolute())

            if not exists(self.directory):
                makedirs(self.directory)

    def join(self, entity: str, extension: Optional[str] = None) -> str:
        """
        Joins the file name with the starting directory and the extension.

        Arguments
        ---------
        `file_name`: the file name that should be joined with the directory.
        `extension`: the extension that should be appended to the generated
        directory.

        Returns
        -------
        A string of the combined path of the directory, file name, and the
        extension.

        Note
        ----
        If no extension is provided, the default extension that been provided
        during class initialization will be used. If there is no default
        extension also, it assumes that the extension is part of the `file_name`
        argument, or it is a directory.
        """

        if extension is None:
            if self.default_extension is None:
                return join(self.directory, entity)
            else:
                f = entity + "." + self.default_extension
                return join(self.directory, f)
        else:
            f = entity + "." + extension
            return join(self.directory, f)

    def new(self, entity: str, extension: Optional[str] = None) -> Pather:
        return Pather(self.join(entity, extension))

    def contents(
        self, filter: str = "*", include_sub_folders: bool = False
    ) -> List[str]:
        """
        Searches for for directory contents.

        Arguments
        ---------
        - `filter`: a criteria according to which contents will be filtered.

        Returns
        -------
        A list of the full path of the contents.
        """
        if include_sub_folders:
            res = []
            for path, dirs, files in walk(self.directory):
                for f in files:
                    if f.endswith(filter):
                        res.append(join(path, f))
            return res
        else:
            import glob

            return glob.glob(
                join(self.directory, filter), recursive=include_sub_folders
            )

    def deeper(self, *dirs_hierarchy) -> Pather:
        """
        Creates a new `Pather` object, of the same default extension, for
        a deeper directories structure.
        """
        return Pather(
            self.directory, *dirs_hierarchy, default_extension=self.default_extension
        )

    def parent(self) -> Pather:
        if self._is_file:
            return Pather(self.directory)
        else:
            return Pather(str(self._path_obj.parent.absolute()))

    def set_extension(self, extension: str) -> None:
        p = self._path_obj.with_suffix(extension)
        self.extension = extension
        self.full_name = self.name + "." + extension

    def exists(self) -> bool:
        return self._path_obj.exists()
