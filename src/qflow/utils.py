"""工具函数模块"""

import os
import yaml
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
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


def set_task_status(task_path: str, status: str, config: dict, error_msg: str = None):
    """
    设置任务状态
    status: 'running', 'success', 'failed'
    """
    status_files = config['status_files']
    task_dir = Path(task_path)

    # 删除旧的状态文件
    for s in ['running', 'success', 'failed']:
        status_file = task_dir / status_files[s]
        try:
            if status_file.exists():
                status_file.unlink()
        except (FileNotFoundError, PermissionError) as e:
            # 文件可能被其他进程删除或无权限
            print(f"Warning: Failed to remove status file {status_file}: {e}")
            pass

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
    返回: 'opt', 'qha_opt', 'phonon', 'qha'
    """
    path = Path(task_path)

    # 检查是否是QHA体积点的优化任务
    if 'volume_' in str(path) and path.name == 'opt':
        # volume_X.XX/opt 是QHA优化任务
        if 'volume_1.0' not in str(path):
            return 'qha_opt'
        # volume_1.0/opt 不应该存在，但如果存在就当作普通opt
        return 'opt'

    if path.name == 'opt' or '/opt' in str(path):
        return 'opt'

    # 区分普通声子和QHA声子任务
    if 'volume_' in str(path) and 'task.' in str(path):
        # 检查是否是volume_1.0（普通声子）还是其他体积点（QHA）
        if 'volume_1.0' in str(path):
            return 'phonon'
        else:
            return 'qha'

    if path.name == 'qha' or '/qha' in str(path):
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
