"""工具函数模块"""

import os
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Iterable, Union


def load_config(config_path: str = None) -> dict:
    """加载配置文件，优先级: 参数 > 环境变量 QFLOW_CONFIG > config.yaml"""
    if config_path is None:
        config_path = os.environ.get('QFLOW_CONFIG', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_task_status(task_path: str, config: dict) -> str:
    """
    获取任务状态
    返回: 'running', 'success', 'failed', 或 'pending'（无状态文件）
    """
    status_files = config['status_files']
    task_dir = Path(task_path)

    if (task_dir / status_files['success']).exists():
        return 'success'
    if (task_dir / status_files['failed']).exists():
        return 'failed'
    if (task_dir / status_files['running']).exists():
        return 'running'
    return 'pending'


def clear_task_status(task_path: Union[str, Path], config: dict,
                      statuses: Iterable[str] = None, remove_error_log: bool = False) -> int:
    """清理任务状态文件。

    Returns:
        实际删除的状态文件数量。
    """
    status_files = config['status_files']
    task_dir = Path(task_path)
    statuses = tuple(statuses or ('running', 'success', 'failed'))

    removed = 0
    for status in statuses:
        status_name = status_files.get(status)
        if not status_name:
            continue

        status_file = task_dir / status_name
        try:
            if status_file.exists():
                status_file.unlink()
                removed += 1
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Failed to remove status file {status_file}: {e}")

    if remove_error_log:
        error_file = task_dir / config['failure']['task_error_file']
        try:
            if error_file.exists():
                error_file.unlink()
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Failed to remove error file {error_file}: {e}")

    return removed


def set_task_status(task_path: str, status: str, config: dict, error_msg: str = None):
    """
    设置任务状态
    status: 'running', 'success', 'failed'
    """
    status_files = config['status_files']
    task_dir = Path(task_path)

    # 删除旧的状态文件
    clear_task_status(task_dir, config, statuses=['running', 'success', 'failed'])

    # 创建新的状态文件
    if status in status_files:
        (task_dir / status_files[status]).touch()

    # 如果是失败状态，记录错误信息
    if status == 'failed' and error_msg:
        error_file = task_dir / config['failure']['task_error_file']
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}\n{error_msg}\n")


def record_failed_task(task_path: str, config: dict):
    """记录失败任务路径到全局失败日志"""
    log_file = Path(config['failure']['log_file'])
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} | {task_path}\n")


def get_task_type(task_path: str) -> str:
    """
    根据任务路径判断任务类型
    返回: 'opt', 'qha_opt', 'phonon', 'qha', 'bte_fc2', 'bte_fc3'
    """
    path = Path(task_path)
    path_str = str(path)

    # BTE 任务
    if '/bte/fc2/' in path_str and 'task.' in path_str:
        return 'bte_fc2'
    if '/bte/fc3/' in path_str and 'task.' in path_str:
        return 'bte_fc3'

    # BTE 压强点优化: P_XXGPa/opt
    if re.match(r'.*P_\d+GPa/opt$', path_str):
        return 'bte_opt'

    # 检查是否是QHA体积点的优化任务
    if 'volume_' in path_str and path.name == 'opt':
        if 'volume_1.0' not in path_str:
            return 'qha_opt'
        return 'opt'

    if path.name == 'opt' or '/opt' in path_str:
        return 'opt'

    # 区分普通声子和QHA声子任务
    if 'volume_' in path_str and 'task.' in path_str:
        if 'volume_1.0' in path_str:
            return 'phonon'
        else:
            return 'qha'

    if path.name == 'qha' or '/qha' in path_str:
        return 'qha'

    return 'unknown'


def get_structure_name(task_path: str, config: dict) -> str:
    """从任务路径中提取结构名称"""
    structures_dir = Path(config['manager']['structures_dir']).resolve()
    task_path = Path(task_path).resolve()

    try:
        rel_path = task_path.relative_to(structures_dir)
        return rel_path.parts[0]  # 第一级目录就是结构名
    except ValueError:
        return None
