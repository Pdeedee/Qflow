"""任务数据库管理 - SQLite实现"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager

from .utils import load_config, get_task_type


class TaskDB:
    """SQLite任务数据库管理器"""

    # 任务优先级定义（数字越大优先级越高）
    PRIORITY = {
        'opt': 100,
        'bte_opt': 90,   # BTE 压强点优化（PSTRESS）
        'qha_opt': 80,   # QHA体积点优化（ISIF=2）
        'phonon': 50,
        'bte_fc2': 45,   # BTE fc2 位移单点
        'bte_fc3': 40,   # BTE fc3 位移单点
        'qha': 30
    }

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.config = config

        work_dir = Path(config.get('work_dir', '.')).resolve()
        self.db_path = work_dir / 'tasks.db'
        self.backup_path = work_dir / 'tasks.db.backup'
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    slurm_job_id TEXT,
                    error_message TEXT
                )
            ''')
            # 创建索引加速查询
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC, created_at ASC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)')
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def backup(self) -> bool:
        """备份数据库（覆盖旧备份）"""
        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, self.backup_path)
                return True
        except Exception as e:
            print(f"[DB] 备份失败: {e}")
        return False

    def restore_from_backup(self) -> bool:
        """从备份恢复数据库"""
        try:
            if self.backup_path.exists():
                shutil.copy2(self.backup_path, self.db_path)
                print(f"[DB] 已从备份恢复数据库")
                return True
        except Exception as e:
            print(f"[DB] 恢复失败: {e}")
        return False

    def add_task(self, task_path: str, task_type: str = None) -> bool:
        """添加任务，如果已存在则跳过"""
        if task_type is None:
            task_type = get_task_type(task_path)

        priority = self.PRIORITY.get(task_type, 0)
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            try:
                conn.execute('''
                    INSERT INTO tasks (path, task_type, status, priority, created_at, updated_at)
                    VALUES (?, ?, 'pending', ?, ?, ?)
                ''', (task_path, task_type, priority, now, now))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # 任务已存在
                return False

    def get_pending_task(self) -> Optional[Dict]:
        """获取优先级最高的pending任务并标记为running"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            # 使用事务保证原子性
            cursor = conn.execute('''
                SELECT * FROM tasks
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            ''')
            row = cursor.fetchone()

            if row is None:
                return None

            # 标记为running
            conn.execute('''
                UPDATE tasks SET status = 'running', updated_at = ?
                WHERE id = ?
            ''', (now, row['id']))
            conn.commit()

            return dict(row)

    def update_status(self, task_path: str, status: str,
                      slurm_job_id: str = None, error_message: str = None):
        """更新任务状态"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            if slurm_job_id:
                conn.execute('''
                    UPDATE tasks SET status = ?, updated_at = ?, slurm_job_id = ?
                    WHERE path = ?
                ''', (status, now, slurm_job_id, task_path))
            elif error_message:
                conn.execute('''
                    UPDATE tasks SET status = ?, updated_at = ?, error_message = ?
                    WHERE path = ?
                ''', (status, now, error_message, task_path))
            else:
                conn.execute('''
                    UPDATE tasks SET status = ?, updated_at = ?
                    WHERE path = ?
                ''', (status, now, task_path))
            conn.commit()

    def update_task_time(self, task_path: str, start_time: str, end_time: str,
                         duration_seconds: float, status: str):
        """更新任务执行时间"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute('''
                UPDATE tasks
                SET start_time = ?, end_time = ?, duration_seconds = ?,
                    status = ?, updated_at = ?
                WHERE path = ?
            ''', (start_time, end_time, duration_seconds, status, now, task_path))
            conn.commit()

    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """获取任务统计"""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT task_type, status, COUNT(*) as count
                FROM tasks
                GROUP BY task_type, status
            ''')

            stats = {}
            for row in cursor:
                task_type = row['task_type']
                status = row['status']
                count = row['count']

                if task_type not in stats:
                    stats[task_type] = {'pending': 0, 'running': 0, 'success': 0, 'failed': 0}
                stats[task_type][status] = count

            return stats

    def get_running_tasks(self) -> List[Dict]:
        """获取所有running任务"""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT * FROM tasks WHERE status = 'running'
                ORDER BY updated_at DESC
            ''')
            return [dict(row) for row in cursor]

    def get_running_count(self) -> int:
        """获取running任务数量"""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', ('running',))
            return cursor.fetchone()[0]

    def get_tasks(self, status: str = None, task_type: str = None,
                  limit: int = None) -> List[Dict]:
        """获取任务列表"""
        query = 'SELECT * FROM tasks WHERE 1=1'
        params = []

        if status:
            query += ' AND status = ?'
            params.append(status)
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)

        query += ' ORDER BY updated_at DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]

    def get_recent_completed(self, hours: int = 6) -> List[Dict]:
        """获取最近完成的任务（用于计算平均时间）"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT * FROM tasks
                WHERE status = 'success' AND end_time > ? AND duration_seconds IS NOT NULL
                ORDER BY end_time DESC
            ''', (cutoff,))
            return [dict(row) for row in cursor]

    def reset_running_tasks(self) -> int:
        """将所有running任务重置为pending"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                UPDATE tasks SET status = 'pending', updated_at = ?
                WHERE status = 'running'
            ''', (now,))
            conn.commit()
            return cursor.rowcount

    def reset_failed_tasks(self) -> int:
        """将所有failed任务重置为pending"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                UPDATE tasks SET status = 'pending', updated_at = ?, error_message = NULL
                WHERE status = 'failed'
            ''', (now,))
            conn.commit()
            return cursor.rowcount

    def reset_task_to_pending(self, task_path: str) -> bool:
        """将指定任务重置为pending状态"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                UPDATE tasks SET status = 'pending', updated_at = ?, error_message = NULL
                WHERE path = ?
            ''', (now, task_path))
            conn.commit()
            return cursor.rowcount > 0

    def remove_task(self, task_path: str) -> bool:
        """从数据库中删除指定任务"""
        with self._get_conn() as conn:
            cursor = conn.execute('DELETE FROM tasks WHERE path = ?', (task_path,))
            conn.commit()
            return cursor.rowcount > 0

    def remove_nonexistent_tasks(self, existing_paths: set) -> int:
        """删除不存在的任务"""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT path FROM tasks')
            all_paths = [row['path'] for row in cursor]

            removed = 0
            for path in all_paths:
                if path not in existing_paths:
                    conn.execute('DELETE FROM tasks WHERE path = ?', (path,))
                    removed += 1

            conn.commit()
            return removed

    def sync_from_filesystem(self, work_dir: Path, structures_dir: Path) -> Dict[str, int]:
        """从文件系统同步任务状态"""
        synced = {'added': 0, 'updated': 0, 'removed': 0}
        existing_paths = set()

        if not structures_dir.exists():
            return synced

        for struct_dir in structures_dir.iterdir():
            if not struct_dir.is_dir():
                continue

            # opt任务
            opt_dir = struct_dir / 'opt'
            if opt_dir.exists() and (opt_dir / 'POSCAR').exists():
                task_path = str(opt_dir.relative_to(work_dir))
                existing_paths.add(task_path)

                status = self._get_fs_status(opt_dir)
                if self.add_task(task_path, 'opt'):
                    synced['added'] += 1
                self.update_status(task_path, status)
                synced['updated'] += 1

            # phonon/qha任务
            for volume_dir in struct_dir.glob('volume_*'):
                for task_dir in volume_dir.glob('task.*'):
                    if not task_dir.is_dir() or task_dir.name == 'task_perfect':
                        continue
                    if not (task_dir / 'POSCAR').exists():
                        continue

                    task_path = str(task_dir.relative_to(work_dir))
                    existing_paths.add(task_path)

                    task_type = 'qha' if volume_dir.name != 'volume_1.0' else 'phonon'
                    status = self._get_fs_status(task_dir)

                    if self.add_task(task_path, task_type):
                        synced['added'] += 1
                    self.update_status(task_path, status)
                    synced['updated'] += 1

        # 删除不存在的任务
        synced['removed'] = self.remove_nonexistent_tasks(existing_paths)

        return synced

    def _get_fs_status(self, task_path: Path) -> str:
        """从文件系统获取任务状态"""
        if (task_path / '.success').exists():
            return 'success'
        if (task_path / '.failed').exists():
            return 'failed'
        if (task_path / '.running').exists():
            return 'running'
        return 'pending'
