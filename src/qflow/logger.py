"""qflow日志模块"""

import logging
from pathlib import Path


def setup_logger(work_dir: str = None):
    """设置qflow logger

    Args:
        work_dir: 工作目录，如果为None则从配置文件读取
    """
    logger = logging.getLogger('qflow')

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 如果没有指定work_dir，尝试从配置读取
    if work_dir is None:
        try:
            from .utils import load_config
            config = load_config()
            work_dir = Path(config.get('work_dir', '.')).resolve()
        except Exception:
            work_dir = Path('.').resolve()
    else:
        work_dir = Path(work_dir).resolve()

    # 格式化器 - 包含时间戳
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    logfile = work_dir / "qflow.log"
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 默认logger实例
logger = setup_logger()
