"""任务数据库管理 - SQLite实现"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Callable, Iterable, Optional, List, Dict
from contextlib import contextmanager

from .utils import load_config, get_task_type, parse_task_metadata


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
        'qha': 30,
        'plain': 20,
    }

    def __init__(self, config: dict = None, skip_backfill: bool = False):
        if config is None:
            config = load_config()
        self.config = config
        self.skip_backfill = skip_backfill

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
                    structure_name TEXT,
                    volume_name TEXT,
                    pressure_name TEXT,
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
            self._ensure_column(conn, 'tasks', 'structure_name', 'TEXT')
            self._ensure_column(conn, 'tasks', 'volume_name', 'TEXT')
            self._ensure_column(conn, 'tasks', 'pressure_name', 'TEXT')

            # 创建索引加速查询
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC, created_at ASC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_structure_name ON tasks(structure_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_volume_name ON tasks(volume_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_pressure_name ON tasks(pressure_name)')
            conn.commit()

        if not self.skip_backfill:
            self._backfill_task_metadata()

    def _ensure_column(self, conn, table_name: str, column_name: str, column_def: str):
        """确保表包含指定列。"""
        cursor = conn.execute(f'PRAGMA table_info({table_name})')
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}')

    def _task_metadata(self, task_path: str) -> Dict[str, Optional[str]]:
        """解析任务元数据。"""
        return parse_task_metadata(task_path, self.config)

    def _backfill_task_metadata(self):
        """为已有任务回填结构/体积/压强元数据。"""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT path FROM tasks
                WHERE structure_name IS NULL OR volume_name IS NULL OR pressure_name IS NULL
            ''')
            rows = cursor.fetchall()
            for row in rows:
                task_path = row['path']
                metadata = self._task_metadata(task_path)
                conn.execute('''
                    UPDATE tasks
                    SET structure_name = ?, volume_name = ?, pressure_name = ?
                    WHERE path = ?
                ''', (
                    metadata['structure_name'],
                    metadata['volume_name'],
                    metadata['pressure_name'],
                    task_path,
                ))
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

    def _build_task_insert_row(self, task_path: str, task_type: str = None,
                               metadata: Dict[str, Optional[str]] = None,
                               now: str = None):
        """构造 tasks 表的插入行。"""
        if task_type is None:
            task_type = get_task_type(task_path)
        if metadata is None:
            metadata = self._task_metadata(task_path)
        if now is None:
            now = datetime.now().isoformat()

        priority = self.PRIORITY.get(task_type, 0)
        return (
            task_path,
            task_type,
            metadata['structure_name'],
            metadata['volume_name'],
            metadata['pressure_name'],
            priority,
            now,
            now,
        )

    def add_task(self, task_path: str, task_type: str = None) -> bool:
        """添加任务，如果已存在则跳过"""
        row = self._build_task_insert_row(task_path, task_type=task_type)

        with self._get_conn() as conn:
            try:
                conn.execute('''
                    INSERT INTO tasks (
                        path, task_type, structure_name, volume_name, pressure_name,
                        status, priority, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ''', row)
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def _insert_task_rows_ignore_existing(self, conn, rows) -> Dict[str, int]:
        """批量写入任务行，仅插入新任务。"""
        if not rows:
            return {'added': 0, 'existing': 0}

        before_changes = conn.total_changes
        conn.executemany('''
            INSERT OR IGNORE INTO tasks (
                path, task_type, structure_name, volume_name, pressure_name,
                status, priority, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        ''', rows)
        conn.commit()

        added = conn.total_changes - before_changes
        return {'added': added, 'existing': len(rows) - added}

    def add_tasks_ignore_existing(self, task_records: Iterable[Dict], batch_size: int = 10000,
                                  progress_callback: Optional[Callable[[int, int, int], None]] = None) -> Dict[str, int]:
        """批量添加任务，仅插入新任务，不修改已有记录。"""
        totals = {'added': 0, 'existing': 0}
        now = datetime.now().isoformat()
        rows = []
        processed = 0

        def build_row(record):
            task_path = record['path']
            task_type = record.get('task_type')
            metadata = {
                'structure_name': record.get('structure_name'),
                'volume_name': record.get('volume_name'),
                'pressure_name': record.get('pressure_name'),
            }
            if task_type is None or any(value is None for value in metadata.values()):
                parsed_metadata = self._task_metadata(task_path)
                metadata = {
                    'structure_name': metadata['structure_name'] or parsed_metadata['structure_name'],
                    'volume_name': metadata['volume_name'] or parsed_metadata['volume_name'],
                    'pressure_name': metadata['pressure_name'] or parsed_metadata['pressure_name'],
                }

            return self._build_task_insert_row(
                task_path,
                task_type=task_type,
                metadata=metadata,
                now=now,
            )

        with self._get_conn() as conn:
            for record in task_records:
                rows.append(build_row(record))
                processed += 1
                if len(rows) < batch_size:
                    continue

                counts = self._insert_task_rows_ignore_existing(conn, rows)
                totals['added'] += counts['added']
                totals['existing'] += counts['existing']
                if progress_callback is not None:
                    progress_callback(processed, totals['added'], totals['existing'])
                rows = []

            if rows:
                counts = self._insert_task_rows_ignore_existing(conn, rows)
                totals['added'] += counts['added']
                totals['existing'] += counts['existing']
                if progress_callback is not None:
                    progress_callback(processed, totals['added'], totals['existing'])

        return totals

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

    def update_status_bulk(self, task_updates: List[tuple]):
        """批量更新任务状态。

        task_updates: [(task_path, status), ...]
        """
        if not task_updates:
            return 0

        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.executemany('''
                UPDATE tasks
                SET status = ?, updated_at = ?
                WHERE path = ?
            ''', [(status, now, task_path) for task_path, status in task_updates])
            conn.commit()
            return len(task_updates)

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
                  structure_name: str = None, volume_name: str = None,
                  pressure_name: str = None, limit: int = None) -> List[Dict]:
        """获取任务列表"""
        query = 'SELECT * FROM tasks WHERE 1=1'
        params = []

        if status:
            query += ' AND status = ?'
            params.append(status)
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)
        if structure_name:
            query += ' AND structure_name = ?'
            params.append(structure_name)
        if volume_name:
            query += ' AND volume_name = ?'
            params.append(volume_name)
        if pressure_name:
            query += ' AND pressure_name = ?'
            params.append(pressure_name)

        query += ' ORDER BY updated_at DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]

    def get_tasks_by_prefix(self, path_prefix: str, status: str = None,
                            task_type: str = None, limit: int = None) -> List[Dict]:
        """按路径前缀获取任务列表。"""
        query = 'SELECT * FROM tasks WHERE path LIKE ?'
        params = [f'{path_prefix}%']

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

    def get_tasks_by_context(self, structure_name: str, volume_name: str = None,
                             pressure_name: str = None, task_type: str = None,
                             status: str = None, limit: int = None) -> List[Dict]:
        """按结构/体积/压强上下文获取任务。"""
        return self.get_tasks(
            status=status,
            task_type=task_type,
            structure_name=structure_name,
            volume_name=volume_name,
            pressure_name=pressure_name,
            limit=limit,
        )

    def get_task(self, task_path: str) -> Optional[Dict]:
        """获取单个任务"""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT * FROM tasks WHERE path = ?', (task_path,))
            row = cursor.fetchone()
            return dict(row) if row else None

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
                UPDATE tasks
                SET status = 'pending',
                    updated_at = ?,
                    start_time = NULL,
                    end_time = NULL,
                    duration_seconds = NULL,
                    slurm_job_id = NULL,
                    error_message = NULL
                WHERE status = 'running'
            ''', (now,))
            conn.commit()
            return cursor.rowcount

    def reset_failed_tasks(self) -> int:
        """将所有failed任务重置为pending"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                UPDATE tasks
                SET status = 'pending',
                    updated_at = ?,
                    start_time = NULL,
                    end_time = NULL,
                    duration_seconds = NULL,
                    slurm_job_id = NULL,
                    error_message = NULL
                WHERE status = 'failed'
            ''', (now,))
            conn.commit()
            return cursor.rowcount

    def reset_task_to_pending(self, task_path: str) -> bool:
        """将指定任务重置为pending状态"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute('''
                UPDATE tasks
                SET status = 'pending',
                    updated_at = ?,
                    start_time = NULL,
                    end_time = NULL,
                    duration_seconds = NULL,
                    slurm_job_id = NULL,
                    error_message = NULL
                WHERE path = ?
            ''', (now, task_path))
            conn.commit()
            return cursor.rowcount > 0

    def reset_tasks_to_pending_bulk(self, task_paths: List[str]) -> int:
        """批量将指定任务重置为pending状态。"""
        if not task_paths:
            return 0

        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.executemany('''
                UPDATE tasks
                SET status = 'pending',
                    updated_at = ?,
                    start_time = NULL,
                    end_time = NULL,
                    duration_seconds = NULL,
                    slurm_job_id = NULL,
                    error_message = NULL
                WHERE path = ?
            ''', [(now, task_path) for task_path in task_paths])
            conn.commit()
            return len(task_paths)

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
