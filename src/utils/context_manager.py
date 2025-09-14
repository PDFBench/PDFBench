import contextlib
import os
import sys
import warnings


@contextlib.contextmanager
def suppress_stdout():
    """上下文管理器，用于抑制标准输出（stdout）"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_stderr():
    """上下文管理器，用于抑制标准错误输出（stderr）"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextlib.contextmanager
def suppress_all_output():
    """上下文管理器，用于同时抑制标准输出和标准错误输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextlib.contextmanager
def suppress_warnings():
    """上下文管理器，用于抑制警告信息"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
