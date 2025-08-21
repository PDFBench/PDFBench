import logging
import os
from functools import lru_cache


class _Logger(logging.Logger):
    r"""A logger that supports rank0 logging."""

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_rank0_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)


def get_logger(name: str | None = None) -> "_Logger":
    return logging.getLogger(name)  # type: ignore


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0  # type: ignore
logging.Logger.warning_rank0 = warning_rank0  # type: ignore
logging.Logger.warning_rank0_once = warning_rank0_once  # type: ignore


def set_global_logger(log_level: int = logging.INFO):
    # TODO: support rich logging
    # import rich
    # rich_handler = RichHandler(
    #     level=log_level,
    #     show_time=True,
    #     show_level=True,
    #     show_path=True,
    #     log_time_format="[%X]", # 类似您原来的 datefmt
    #     markup=True, # 允许在日志消息中使用 Rich 的标记语言
    #     rich_tracebacks=True, # <-- 关键功能！提供精美的异常回溯
    # )
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [%(levelname)s{'|Rank' + os.environ.get('LOCAL_RANK', '?') if os.environ.get('LOCAL_RANK', '?') != '?' else ''}|%(name)s:%(lineno)s] >> %(message)s",
        handlers=[logging.StreamHandler()],
    )
