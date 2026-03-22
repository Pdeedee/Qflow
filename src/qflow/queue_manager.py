"""队列管理模块 - 使用文件系统队列"""

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .utils import load_config, get_task_type


class QueueManager:
    """任务队列管理器 - 基于文件系统的原子操作"""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.config = config

        # 队列目录结构
        work_dir = Path(config.get('work_dir', '.')).resolve()
        self.queue_dir = work_dir / 'task_queue'
        self.pending_dir = self.queue_dir / 'pending'
        self.running_dir = self.queue_dir / 'running'
        self.done_dir = self.queue_dir / 'done'
        self.failed_dir = self.queue_dir / 'failed'

        self._init_dirs()

    def _init_dirs(self):
        """初始化队列目录"""
        for dir_path in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _task_to_filename(self, task_path: str, task_type: str, priority: int = 0) -> str:
        """
        将任务路径转换为文件名
        格式: {priority:03d}_{timestamp}_{task_type}_{hash}.json
        priority 作为前缀用于排序（数字越大优先级越高）
        """
        # 使用路径的 hash 作为唯一标识
        task_hash = abs(hash(task_path)) % 1000000
        timestamp = int(time.time() * 1000000)  # 微秒级时间戳
        return f"{priority:03d}_{timestamp}_{task_type}_{task_hash:06d}.json"

    def _filename_to_task(self, filename: str, queue_path: Path) -> Optional[Dict]:
        """从文件名和内容解析任务信息"""
        file_path = queue_path / filename
        try:
            with open(file_path, 'r') as f:
                task_data = json.load(f)
            return task_data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def add_task(self, task_path: str, task_type: str = None, priority: int = 0) -> bool:
        """
        添加任务到队列
        返回: True表示添加成功，False表示已存在
        """
        if task_type is None:
            task_type = get_task_type(task_path)

        # 检查任务是否已存在（在任何目录中）
        for queue_dir in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            for f in queue_dir.glob('*.json'):
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        if data['path'] == task_path:
                            return False  # 任务已存在
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        # 创建任务文件
        filename = self._task_to_filename(task_path, task_type, priority)
        file_path = self.pending_dir / filename

        task_data = {
            'path': task_path,
            'task_type': task_type,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(task_data, f)

        return True

    def add_tasks(self, tasks: List[tuple]) -> int:
        """
        批量添加任务
        tasks: [(path, task_type), ...] 或 [(path, task_type, priority), ...]
        返回: 成功添加的数量
        """
        added = 0
        for task_info in tasks:
            if len(task_info) == 2:
                path, task_type = task_info
                priority = 0
            else:
                path, task_type, priority = task_info

            if self.add_task(path, task_type, priority):
                added += 1
        return added

    def get_pending_task(self) -> Optional[Dict]:
        """
        获取一个pending状态的任务并标记为running
        使用 os.rename() 的原子性保证只有一个 worker 能抢到
        优先级高的任务优先执行（文件名前缀数字越大越优先）
        同优先级内，先生成的任务先执行（FIFO）
        返回: 任务信息字典，无任务时返回None
        """
        # 列出所有 pending 任务，按优先级和时间戳排序
        # 文件名格式: {priority:03d}_{timestamp}_{task_type}_{hash}.json
        def sort_key(file_path):
            parts = file_path.name.split('_')
            priority = int(parts[0])  # 优先级
            timestamp = int(parts[1])  # 时间戳
            return (-priority, timestamp)  # 优先级倒序，时间戳正序

        pending_files = sorted(self.pending_dir.glob('*.json'), key=sort_key)

        for file_path in pending_files:
            # 尝试原子性地移动文件到 running 目录
            new_path = self.running_dir / file_path.name
            try:
                os.rename(file_path, new_path)
                # 成功抢到任务，读取任务信息
                with open(new_path, 'r') as f:
                    task_data = json.load(f)
                task_data['updated_at'] = datetime.now().isoformat()
                with open(new_path, 'w') as f:
                    json.dump(task_data, f)
                return task_data
            except FileNotFoundError:
                # 文件已被其他 worker 抢走，继续下一个
                continue
            except Exception as e:
                # 其他错误，跳过这个任务
                print(f"Warning: Failed to claim task {file_path.name}: {e}")
                continue

        return None

    def update_task_status(self, task_path: str, status: str):
        """
        更新任务状态（通过移动文件到不同目录）
        status: 'pending', 'running', 'success', 'failed'
        """
        # 状态到目录的映射
        status_to_dir = {
            'pending': self.pending_dir,
            'running': self.running_dir,
            'success': self.done_dir,
            'failed': self.failed_dir
        }

        if status not in status_to_dir:
            raise ValueError(f"Invalid status: {status}")

        target_dir = status_to_dir[status]

        # 在所有目录中查找任务文件
        found = False
        for source_dir in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            if source_dir == target_dir:
                continue

            for file_path in source_dir.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    if task_data['path'] == task_path:
                        # 找到了任务，移动到目标目录
                        found = True
                        task_data['updated_at'] = datetime.now().isoformat()
                        new_path = target_dir / file_path.name

                        # 写入更新后的数据
                        with open(file_path, 'w') as f:
                            json.dump(task_data, f)

                        # 移动文件
                        os.rename(file_path, new_path)
                        print(f"[QUEUE-UPDATE] {task_path}: {source_dir.name} -> {target_dir.name}")
                        return
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"[QUEUE-ERROR] Failed to process {file_path.name}: {e}")
                    continue

        if not found:
            print(f"[QUEUE-WARNING] Task not found in any queue: {task_path}")

    def sync_task_status(self, task_path: str, status: str):
        """同步任务状态（从文件状态同步到队列）"""
        self.update_task_status(task_path, status)

    def update_task_time(self, task_path: str, start_time: str, end_time: str,
                         duration_seconds: float, status: str) -> bool:
        """
        更新任务的执行时间信息到队列数据库

        Args:
            task_path: 任务路径
            start_time: 开始时间 (ISO格式)
            end_time: 结束时间 (ISO格式)
            duration_seconds: 持续时间（秒）
            status: 任务状态

        Returns: 是否更新成功
        """
        # 在所有目录中查找任务文件
        for dir_path in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            for file_path in dir_path.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    if task_data['path'] == task_path:
                        # 找到了任务，更新时间信息
                        task_data['start_time'] = start_time
                        task_data['end_time'] = end_time
                        task_data['duration_seconds'] = duration_seconds
                        task_data['duration_hours'] = duration_seconds / 3600
                        task_data['execution_status'] = status
                        task_data['updated_at'] = datetime.now().isoformat()

                        with open(file_path, 'w') as f:
                            json.dump(task_data, f, indent=2)

                        return True
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        return False

    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        获取任务统计信息
        返回: {task_type: {status: count, ...}, ...}
        """
        stats = {}

        # 状态到目录的映射
        status_mapping = [
            ('pending', self.pending_dir),
            ('running', self.running_dir),
            ('success', self.done_dir),
            ('failed', self.failed_dir)
        ]

        for status, dir_path in status_mapping:
            for file_path in dir_path.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    # 检查任务路径是否存在
                    task_path = Path(task_data['path'])
                    if not task_path.exists():
                        # 任务目录已被删除，删除任务文件
                        print(f"[QUEUE-CLEANUP] Removing task for deleted path: {task_data['path']} (status: {status})")
                        file_path.unlink()
                        continue

                    task_type = task_data['task_type']
                    if task_type not in stats:
                        stats[task_type] = {'pending': 0, 'running': 0, 'success': 0, 'failed': 0}

                    stats[task_type][status] += 1
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        return stats

    def get_running_tasks(self) -> List[Dict]:
        """获取所有正在运行的任务"""
        running_tasks = []

        for file_path in self.running_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)

                # 检查任务路径是否存在
                task_path = Path(task_data['path'])
                if not task_path.exists():
                    # 任务目录已被删除，删除任务文件
                    print(f"[QUEUE-CLEANUP] Removing running task for deleted path: {task_data['path']}")
                    file_path.unlink()
                    continue

                running_tasks.append(task_data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        # 按更新时间倒序排序
        running_tasks.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return running_tasks

    def get_failed_tasks(self, limit: int = 10) -> List[str]:
        """获取最近失败的任务路径"""
        failed_tasks = []

        for file_path in self.failed_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)

                # 检查任务路径是否存在
                task_path = Path(task_data['path'])
                if not task_path.exists():
                    # 任务目录已被删除，删除任务文件
                    print(f"[QUEUE-CLEANUP] Removing failed task for deleted path: {task_data['path']}")
                    file_path.unlink()
                    continue

                failed_tasks.append(task_data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        # 按更新时间倒序排序
        failed_tasks.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

        return [task['path'] for task in failed_tasks[:limit]]

    def get_all_tasks(self, status: str = None, task_type: str = None) -> List[Dict]:
        """获取所有任务（可按状态和类型过滤）"""
        all_tasks = []

        # 目录映射
        status_dirs = {
            'pending': self.pending_dir,
            'running': self.running_dir,
            'success': self.done_dir,
            'failed': self.failed_dir
        }

        # 确定要搜索的目录
        if status:
            dirs_to_search = [(status, status_dirs.get(status))]
        else:
            dirs_to_search = list(status_dirs.items())

        for status_name, dir_path in dirs_to_search:
            if not dir_path:
                continue

            for file_path in dir_path.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    # 检查任务路径是否存在
                    task_path = Path(task_data['path'])
                    if not task_path.exists():
                        # 任务目录已被删除，删除任务文件
                        print(f"[QUEUE-CLEANUP] Removing {status_name} task for deleted path: {task_data['path']}")
                        file_path.unlink()
                        continue

                    # 添加状态信息
                    task_data['status'] = status_name

                    # 过滤任务类型
                    if task_type and task_data.get('task_type') != task_type:
                        continue

                    all_tasks.append(task_data)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        return all_tasks

    def remove_nonexistent_tasks(self, existing_paths: set) -> int:
        """删除队列中文件系统不存在的任务

        Args:
            existing_paths: 文件系统中存在的任务路径集合

        Returns:
            删除的任务数量
        """
        removed = 0

        for dir_path in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            for file_path in dir_path.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    task_path = task_data['path']
                    if task_path not in existing_paths:
                        file_path.unlink()
                        removed += 1
                        print(f"[QUEUE-CLEANUP] Removed non-existent task: {task_path}")
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        return removed

    def reset_running_tasks(self):
        """将所有running状态的任务重置为pending（用于恢复）"""
        count = 0
        for file_path in self.running_dir.glob('*.json'):
            try:
                # 读取任务数据
                with open(file_path, 'r') as f:
                    task_data = json.load(f)

                # 更新时间
                task_data['updated_at'] = datetime.now().isoformat()

                # 写回并移动到 pending
                with open(file_path, 'w') as f:
                    json.dump(task_data, f)

                new_path = self.pending_dir / file_path.name
                os.rename(file_path, new_path)
                count += 1
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return count

    def reset_failed_tasks(self, clean_files: bool = True, high_priority: bool = True) -> int:
        """
        将所有failed状态的任务重置为pending
        clean_files: 是否同时删除任务目录下的.failed文件
        high_priority: 是否设置高优先级（优先执行）
        返回: 重置的任务数量
        """
        count = 0

        for file_path in self.failed_dir.glob('*.json'):
            try:
                # 读取任务数据
                with open(file_path, 'r') as f:
                    task_data = json.load(f)

                # 删除.failed文件（如果需要）
                if clean_files:
                    failed_file = Path(task_data['path']) / '.failed'
                    if failed_file.exists():
                        failed_file.unlink()

                # 设置高优先级
                if high_priority:
                    task_data['priority'] = 10

                task_data['updated_at'] = datetime.now().isoformat()

                # 重新生成文件名（带优先级）
                priority = task_data.get('priority', 0)
                new_filename = self._task_to_filename(
                    task_data['path'],
                    task_data['task_type'],
                    priority
                )
                new_path = self.pending_dir / new_filename

                # 写入数据并移动
                with open(new_path, 'w') as f:
                    json.dump(task_data, f)

                # 删除旧文件
                file_path.unlink()
                count += 1

            except (FileNotFoundError, json.JSONDecodeError):
                continue

        return count

    def recover_timeout_tasks(self, timeout_seconds: int = 3600):
        """
        恢复超时的任务
        将 running 目录中超过 timeout_seconds 未更新的任务移回 pending
        """
        current_time = datetime.now()
        recovered = 0

        for file_path in self.running_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)

                # 检查更新时间
                updated_at_str = task_data.get('updated_at', '')
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    elapsed = (current_time - updated_at).total_seconds()

                    if elapsed > timeout_seconds:
                        # 任务超时，移回 pending

                        # 清理任务目录中的 .running 状态文件
                        running_file = Path(task_data['path']) / '.running'
                        if running_file.exists():
                            running_file.unlink()

                        task_data['updated_at'] = current_time.isoformat()

                        with open(file_path, 'w') as f:
                            json.dump(task_data, f)

                        new_path = self.pending_dir / file_path.name
                        os.rename(file_path, new_path)
                        recovered += 1

            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                continue

        if recovered > 0:
            print(f"Recovered {recovered} timeout tasks")

        return recovered

    def sync_tasks(self) -> Dict[str, int]:
        """
        同步任务队列：
        1. 删除指向不存在路径的任务
        2. 扫描文件系统，添加新的任务
        3. 修正任务类型（区分phonon和qha）
        返回: {'removed': count, 'added': count, 'updated': count}
        """
        removed = 0
        added = 0
        updated = 0

        # 1. 清理不存在的任务，同时修正任务类型
        print("扫描队列中的任务...")
        status_dirs = [
            ('pending', self.pending_dir),
            ('running', self.running_dir),
            ('success', self.done_dir),
            ('failed', self.failed_dir)
        ]

        for status_name, dir_path in status_dirs:
            for file_path in dir_path.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)

                    task_path = Path(task_data['path'])

                    # 检查路径是否存在
                    if not task_path.exists():
                        print(f"  删除: {task_data['path']} (不存在)")
                        file_path.unlink()
                        removed += 1
                        continue

                    # 修正任务类型
                    correct_type = get_task_type(task_data['path'])
                    if task_data['task_type'] != correct_type:
                        print(f"  更新类型: {task_data['path']} ({task_data['task_type']} -> {correct_type})")
                        task_data['task_type'] = correct_type
                        task_data['updated_at'] = datetime.now().isoformat()

                        # 重新生成文件名（包含正确的task_type）
                        old_file = file_path
                        new_filename = self._task_to_filename(
                            task_data['path'],
                            task_data['task_type'],
                            task_data.get('priority', 0)
                        )
                        new_file = dir_path / new_filename

                        # 写入更新后的数据
                        with open(new_file, 'w') as f:
                            json.dump(task_data, f)

                        # 删除旧文件（如果文件名不同）
                        if old_file != new_file:
                            old_file.unlink()

                        updated += 1

                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        # 2. 扫描并添加新任务
        print("\n扫描文件系统中的新任务...")
        work_dir = Path(self.config.get('work_dir', '.')).resolve()

        # 扫描 structures 目录下的 opt 任务
        structures_dir = work_dir / 'structures'
        if structures_dir.exists():
            for mp_dir in structures_dir.glob('mp-*'):
                if not mp_dir.is_dir():
                    continue
                opt_dir = mp_dir / 'opt'
                if opt_dir.exists() and opt_dir.is_dir():
                    task_path = str(opt_dir.relative_to(work_dir))
                    if self.add_task(task_path, 'opt', priority=100):
                        added += 1
                        print(f"  添加: {task_path}")

        # 扫描 qha_structures 目录
        qha_dir = work_dir / 'qha_structures'
        if qha_dir.exists():
            # opt 任务
            for mp_dir in qha_dir.glob('mp-*'):
                if not mp_dir.is_dir():
                    continue
                opt_dir = mp_dir / 'opt'
                if opt_dir.exists() and opt_dir.is_dir():
                    task_path = str(opt_dir.relative_to(work_dir))
                    if self.add_task(task_path, 'opt', priority=100):
                        added += 1
                        print(f"  添加: {task_path}")

            # phonon 任务
            for mp_dir in qha_dir.glob('mp-*'):
                if not mp_dir.is_dir():
                    continue
                for volume_dir in mp_dir.glob('volume_*'):
                    if not volume_dir.is_dir():
                        continue
                    for task_dir in volume_dir.glob('task.*'):
                        if task_dir.is_dir():
                            task_path = str(task_dir.relative_to(work_dir))
                            if self.add_task(task_path, 'phonon', priority=50):
                                added += 1
                                print(f"  添加: {task_path}")

        return {'removed': removed, 'added': added, 'updated': updated}
