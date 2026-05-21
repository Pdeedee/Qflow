"""任务数据库管理 - SQLite实现"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Callable, Iterable, Optional, List, Dict
from contextlib import contextmanager

from .utils import load_config, get_task_type, parse_task_metadata


class ImaginaryFrequencyDB:
    """虚频检查缓存数据库。"""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.config = config
        self.work_dir = Path(config.get('work_dir', '.')).resolve()
        self.db_path = self.work_dir / 'tasks.db'
        self._init_db()

    def _init_db(self):
        """初始化虚频缓存表。"""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS imaginary_frequency_cache (
                    volume_path TEXT PRIMARY KEY,
                    phonopy_params_path TEXT NOT NULL,
                    phonopy_params_mtime_ns INTEGER NOT NULL,
                    phonopy_params_size INTEGER NOT NULL,
                    has_imaginary INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_imaginary_updated_at '
                'ON imaginary_frequency_cache(updated_at)'
            )
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接。"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _normalize_path(self, path: Path) -> str:
        """规范化路径，优先保存为相对 work_dir 的路径。"""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self.work_dir))
        except ValueError:
            return str(resolved)

    def _build_signature(self, volume_dir: Path) -> Optional[Dict[str, object]]:
        """构造用于缓存校验的 phonopy_params 签名。"""
        phonopy_params = Path(volume_dir) / 'analyze' / 'phonopy_params.yaml'
        if not phonopy_params.exists():
            return None

        stat = phonopy_params.stat()
        return {
            'volume_path': self._normalize_path(Path(volume_dir)),
            'phonopy_params_path': self._normalize_path(phonopy_params),
            'phonopy_params_mtime_ns': stat.st_mtime_ns,
            'phonopy_params_size': stat.st_size,
        }

    def get_cached_result(self, volume_dir: Path) -> Optional[bool]:
        """获取缓存的虚频判断结果；缓存失效时返回 None。"""
        volume_path = self._normalize_path(Path(volume_dir))

        with self._get_conn() as conn:
            row = conn.execute(
                '''
                SELECT
                    phonopy_params_path,
                    phonopy_params_mtime_ns,
                    phonopy_params_size,
                    has_imaginary
                FROM imaginary_frequency_cache
                WHERE volume_path = ?
                ''',
                (volume_path,),
            ).fetchone()

        if row is None:
            return None

        signature = self._build_signature(Path(volume_dir))
        if signature is None:
            self.invalidate(volume_dir)
            return None

        if (
            row['phonopy_params_path'] != signature['phonopy_params_path']
            or row['phonopy_params_mtime_ns'] != signature['phonopy_params_mtime_ns']
            or row['phonopy_params_size'] != signature['phonopy_params_size']
        ):
            self.invalidate(volume_dir)
            return None

        return bool(row['has_imaginary'])

    def set_cached_result(self, volume_dir: Path, has_imaginary: bool) -> bool:
        """写入虚频判断结果缓存。"""
        signature = self._build_signature(Path(volume_dir))
        if signature is None:
            self.invalidate(volume_dir)
            return False

        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                '''
                INSERT INTO imaginary_frequency_cache (
                    volume_path,
                    phonopy_params_path,
                    phonopy_params_mtime_ns,
                    phonopy_params_size,
                    has_imaginary,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(volume_path) DO UPDATE SET
                    phonopy_params_path = excluded.phonopy_params_path,
                    phonopy_params_mtime_ns = excluded.phonopy_params_mtime_ns,
                    phonopy_params_size = excluded.phonopy_params_size,
                    has_imaginary = excluded.has_imaginary,
                    updated_at = excluded.updated_at
                ''',
                (
                    signature['volume_path'],
                    signature['phonopy_params_path'],
                    signature['phonopy_params_mtime_ns'],
                    signature['phonopy_params_size'],
                    int(has_imaginary),
                    now,
                ),
            )
            conn.commit()
        return True

    def invalidate(self, volume_dir: Path):
        """删除指定体积目录的缓存。"""
        volume_path = self._normalize_path(Path(volume_dir))
        with self._get_conn() as conn:
            conn.execute(
                'DELETE FROM imaginary_frequency_cache WHERE volume_path = ?',
                (volume_path,),
            )
            conn.commit()


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
        'bte_postprocess': 35,  # BTE 热导率后处理
        'qha': 30,
        'plain': 20,
    }

    STATUS_ALIASES = {
        'timeout': 'failed',
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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS workflow_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    structure_name TEXT NOT NULL,
                    volume_name TEXT NOT NULL DEFAULT '',
                    pressure_name TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'done',
                    source_task TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(structure_name, volume_name, pressure_name, stage)
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_stage ON workflow_state(stage)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_state ON workflow_state(state)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_structure ON workflow_state(structure_name)')
            self._normalize_legacy_statuses(conn)
            conn.commit()

        if not self.skip_backfill:
            self._backfill_task_metadata()
            self.backfill_workflow_states_from_tasks()

    def _ensure_column(self, conn, table_name: str, column_name: str, column_def: str):
        """确保表包含指定列。"""
        cursor = conn.execute(f'PRAGMA table_info({table_name})')
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}')

    def _task_metadata(self, task_path: str) -> Dict[str, Optional[str]]:
        """解析任务元数据。"""
        return parse_task_metadata(task_path, self.config)

    def _normalize_status(self, status: Optional[str]) -> Optional[str]:
        """规范化任务状态，兼容历史别名。"""
        if status is None:
            return None
        return self.STATUS_ALIASES.get(status, status)

    def _normalize_legacy_statuses(self, conn):
        """将历史状态值迁移到当前状态集合。"""
        for old_status, new_status in self.STATUS_ALIASES.items():
            conn.execute(
                'UPDATE tasks SET status = ? WHERE status = ?',
                (new_status, old_status),
            )

    def _normalize_workflow_value(self, value: Optional[str]) -> str:
        """将 workflow_state 上下文字段规范化为空串。"""
        return value or ''

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

    def _workflow_row(self, structure_name: str, stage: str, state: str = 'done',
                      volume_name: Optional[str] = None, pressure_name: Optional[str] = None,
                      source_task: Optional[str] = None, updated_at: Optional[str] = None):
        """构造 workflow_state 表的插入行。"""
        if updated_at is None:
            updated_at = datetime.now().isoformat()
        return (
            structure_name,
            self._normalize_workflow_value(volume_name),
            self._normalize_workflow_value(pressure_name),
            stage,
            state,
            source_task,
            updated_at,
        )

    def set_workflow_state(self, structure_name: str, stage: str, state: str = 'done',
                           volume_name: Optional[str] = None, pressure_name: Optional[str] = None,
                           source_task: Optional[str] = None):
        """写入或更新 workflow_state。"""
        row = self._workflow_row(
            structure_name,
            stage,
            state=state,
            volume_name=volume_name,
            pressure_name=pressure_name,
            source_task=source_task,
        )
        with self._get_conn() as conn:
            conn.execute(
                '''
                INSERT INTO workflow_state (
                    structure_name, volume_name, pressure_name, stage, state, source_task, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(structure_name, volume_name, pressure_name, stage) DO UPDATE SET
                    state = excluded.state,
                    source_task = excluded.source_task,
                    updated_at = excluded.updated_at
                ''',
                row,
            )
            conn.commit()

    def get_workflow_states(self, stage: Optional[str] = None, state: Optional[str] = None,
                            structure_name: Optional[str] = None, volume_name: Optional[str] = None,
                            pressure_name: Optional[str] = None) -> List[Dict]:
        """查询 workflow_state 记录。"""
        query = 'SELECT * FROM workflow_state WHERE 1=1'
        params = []

        if stage is not None:
            query += ' AND stage = ?'
            params.append(stage)
        if state is not None:
            query += ' AND state = ?'
            params.append(state)
        if structure_name is not None:
            query += ' AND structure_name = ?'
            params.append(structure_name)
        if volume_name is not None:
            query += ' AND volume_name = ?'
            params.append(self._normalize_workflow_value(volume_name))
        if pressure_name is not None:
            query += ' AND pressure_name = ?'
            params.append(self._normalize_workflow_value(pressure_name))

        query += ' ORDER BY updated_at DESC'
        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]

    def has_workflow_state(self, structure_name: str, stage: str, state: Optional[str] = None,
                           volume_name: Optional[str] = None, pressure_name: Optional[str] = None) -> bool:
        """判断 workflow_state 是否存在。"""
        query = '''
            SELECT 1 FROM workflow_state
            WHERE structure_name = ? AND volume_name = ? AND pressure_name = ? AND stage = ?
        '''
        params = [
            structure_name,
            self._normalize_workflow_value(volume_name),
            self._normalize_workflow_value(pressure_name),
            stage,
        ]
        if state is not None:
            query += ' AND state = ?'
            params.append(state)
        query += ' LIMIT 1'

        with self._get_conn() as conn:
            return conn.execute(query, params).fetchone() is not None

    def delete_workflow_states(self, structure_name: Optional[str] = None,
                               volume_name: Optional[str] = None,
                               pressure_name: Optional[str] = None,
                               stages: Optional[Iterable[str]] = None):
        """删除 workflow_state 记录。"""
        query = 'DELETE FROM workflow_state WHERE 1=1'
        params = []

        if structure_name is not None:
            query += ' AND structure_name = ?'
            params.append(structure_name)
        if volume_name is not None:
            query += ' AND volume_name = ?'
            params.append(self._normalize_workflow_value(volume_name))
        if pressure_name is not None:
            query += ' AND pressure_name = ?'
            params.append(self._normalize_workflow_value(pressure_name))
        if stages:
            stage_list = list(stages)
            placeholders = ', '.join('?' for _ in stage_list)
            query += f' AND stage IN ({placeholders})'
            params.extend(stage_list)

        with self._get_conn() as conn:
            conn.execute(query, params)
            conn.commit()

    def backfill_workflow_states_from_tasks(self):
        """根据已有任务回填 workflow_state。"""
        stage_by_type = {
            'opt': 'opt_generated',
            'phonon': 'phonon_generated',
            'qha_opt': 'qha_opt_generated',
            'qha': 'qha_generated',
            'bte_opt': 'bte_opt_generated',
            'bte_fc2': 'bte_fc2_generated',
            'bte_fc3': 'bte_fc3_generated',
            'bte_postprocess': 'bte_postprocess_generated',
            'plain': 'plain_generated',
        }

        with self._get_conn() as conn:
            rows = conn.execute('''
                SELECT DISTINCT task_type, structure_name, volume_name, pressure_name
                FROM tasks
                WHERE structure_name IS NOT NULL
            ''').fetchall()

            now = datetime.now().isoformat()
            workflow_rows = []
            for row in rows:
                stage = stage_by_type.get(row['task_type'])
                if stage is None:
                    continue
                workflow_rows.append(self._workflow_row(
                    row['structure_name'],
                    stage,
                    volume_name=row['volume_name'],
                    pressure_name=row['pressure_name'],
                    updated_at=now,
                ))

            if workflow_rows:
                conn.executemany(
                    '''
                    INSERT INTO workflow_state (
                        structure_name, volume_name, pressure_name, stage, state, source_task, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(structure_name, volume_name, pressure_name, stage) DO NOTHING
                    ''',
                    workflow_rows,
                )
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
        status = self._normalize_status(status)
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

        normalized_updates = [
            (task_path, self._normalize_status(status))
            for task_path, status in task_updates
        ]
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.executemany('''
                UPDATE tasks
                SET status = ?, updated_at = ?
                WHERE path = ?
            ''', [(status, now, task_path) for task_path, status in normalized_updates])
            conn.commit()
            return len(normalized_updates)

    def update_task_time(self, task_path: str, start_time: str, end_time: str,
                         duration_seconds: float, status: str):
        """更新任务执行时间"""
        status = self._normalize_status(status)
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
            status = self._normalize_status(status)
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
            status = self._normalize_status(status)
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

    def reset_success_tasks(self) -> int:
        """将所有success任务重置为pending"""
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
                WHERE status = 'success'
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
                for task_dir in sorted(volume_dir.glob('task.*')):
                    if not task_dir.is_dir():
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
        return 'pending'
