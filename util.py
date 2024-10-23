import os


def get_temp_path(extension: str = "", seed: str = "42"):
    # try to get a nice random name
    try_path = "tmp" + str(abs(hash(seed))) + extension

    if not os.path.exists(try_path):
        return try_path

    # fall back to trying sequential names

    for i in range(4294967296):
        if not os.path.exists("tmp" + str(abs(hash(seed))) + extension):
            return try_path

    raise FileExistsError("Could not find available temp file name. Make sure to clean up temp files after use.")


class TempFile:
    def __init__(self, extension: str = "", seed: str = "42"):
        self.path = get_temp_path(extension, seed)

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        return False

