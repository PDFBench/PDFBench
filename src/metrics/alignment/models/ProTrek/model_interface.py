# This file is copied from [PrtTrek](https://github.com/westlake-repl/ProTrek)
# Original license: MIT License
import os


def register_model(cls):
    global now_cls
    now_cls = cls
    return cls


now_cls = None


class ModelInterface:
    @classmethod
    def init_model(cls, model_py_path: str, **kwargs):
        """

        Args:
            model_py_path: Py file Path of model you want to use.
           **kwargs: Kwargs for model initialization

        Returns: Corresponding model
        """
        sub_dirs = model_py_path.split(os.sep)
        cmd = f"from {'.' + '.'.join(sub_dirs[:-1])} import {sub_dirs[-1]}"
        exec(cmd)

        return now_cls(**kwargs)
