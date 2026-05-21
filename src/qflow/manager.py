"""Manager管理进程 - Sbatch模式，直接提交任务到SLURM"""

import os
import time
import json
import shutil
import subprocess
import re
from pathlib import Path
from typing import Iterable, List, Set, Optional, Dict, Tuple
from datetime import datetime

from .logger import logger

logger.info("正在加载 ASE...")
from ase.io import read, write

logger.info("正在加载 qflow 模块...")
from .utils import (
    load_config,
    get_structure_name,
    clear_task_status,
)
from .template import generate_task_script
from .task_db import TaskDB, ImaginaryFrequencyDB
from .submit_registry import SubmitTaskScanner
from .phonon_utils import (
    generate_phonon_displacements,
    postprocess_phonon,
    check_imaginary_frequency,
    postprocess_qha,
    get_missing_qha_static_energy_volumes,
    generate_bte_displacements,
)

logger.info("正在加载 pymatgen...")
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# VASP输入文件生成
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MatPESStaticSet
from pymatgen.core import Structure as PMGStructure

logger.info("所有模块加载完成")


class Manager:
    """任务管理器 - Sbatch模式"""

    @staticmethod
    def _normalize_volume_list(volumes) -> List[float]:
        """规范化 QHA 体积列表，确保后续可直接做数值计算。"""
        normalized = []
        for raw_volume in volumes:
            try:
                normalized.append(float(raw_volume))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid qha.volumes entry: {raw_volume!r}") from exc
        return normalized

    def __init__(self, config: dict = None):
        logger.info("正在初始化 Manager...")
        if config is None:
            config = load_config()
        self.config = config

        # 基本配置
        self.work_dir = Path(config.get('work_dir', '.')).resolve()
        self.structures_dir = (self.work_dir / config['manager']['structures_dir']).resolve()
        self.scan_interval = config['manager']['scan_interval']

        # 任务时间记录目录
        self.task_times_dir = self.work_dir / 'task_times'
        self.task_times_dir.mkdir(exist_ok=True)

        # Job ID映射文件
        self.jobs_file = self.work_dir / 'sbatch_jobs.json'

        # 最大并发任务数配置文件
        self.max_workers_file = self.work_dir / 'max_workers.txt'

        # 从配置读取默认max_workers
        default_max_workers = config.get('manager', {}).get('max_workers', 25)
        if not self.max_workers_file.exists():
            self.max_workers_file.write_text(str(default_max_workers))
            logger.info(f"设置默认最大并发数: {default_max_workers}")

        # 声子和QHA配置
        self.phonon_config = config.get('phonon', {})
        self.qha_config = config.get('qha', {})
        self.bte_config = config.get('bte', {})

        # Worker配置
        self.worker_mode = config.get('worker', {}).get('mode', 'mattersim')

        # 结构优化配置
        opt_config = config.get('opt', {})
        self.refine_structure = opt_config.get('refine_structure', False)

        # 工作流开关: bte: true/false, qha: true/false
        self.enable_qha = config.get('manager', {}).get('qha', True)
        self.enable_bte = config.get('manager', {}).get('bte', False)
        self.plain_submit = config.get('manager', {}).get('plain_submit', False)

        # INCAR设置在每次提交任务时实时从config.yaml读取（见_generate_vasp_inputs）

        # POTCAR设置
        potcar_config = config.get('potcar', {})
        self.potcar_functional = potcar_config.get('functional', 'PBE_54')

        # 默认体积列表 (用于QHA)
        self.default_volumes = self._normalize_volume_list(
            self.qha_config.get('volumes', [0.98, 0.99, 1.0, 1.01, 1.02])
        )

        # SQLite任务数据库
        logger.info("正在初始化任务数据库...")
        self.db = TaskDB(config, skip_backfill=self.plain_submit)
        self.imaginary_db = ImaginaryFrequencyDB(config)
        self.submit_scanner = SubmitTaskScanner(self.work_dir, self.structures_dir)
        self._backfill_workflow_state_from_markers()

        # 标记已处理的结构，避免重复
        self._phonon_generated: Set[str] = set()
        self._qha_generated: Set[str] = set()
        self._phonon_postprocessed: Set[str] = set()
        self._qha_postprocessed: Set[str] = set()
        self._bte_generated: Set[str] = set()
        self._bte_postprocess_generated: Set[str] = set()
        self._postprocess_failures: Set[str] = set()

        logger.info("Manager 初始化完成")

    # ========== 任务状态管理 ==========

    def get_task_status(self, task_path: Path) -> str:
        """检查任务状态（仅识别 .success，其他状态由数据库维护）"""
        if (task_path / '.success').exists():
            return 'success'
        if (task_path / 'POSCAR').exists():
            return 'pending'
        return 'not_ready'

    def _iter_submit_task_dirs(self, parent_dir: Path) -> List[Path]:
        """返回应被注册/提交的任务目录。"""
        return [task_dir for task_dir in sorted(parent_dir.glob('task.*')) if task_dir.is_dir()]

    def _register_submit_tasks(self, parent_dir: Path, task_type: str) -> List[Path]:
        """确保目录下所有 task.* 都已注册到数据库。"""
        task_dirs = []
        for task_dir in self._iter_submit_task_dirs(parent_dir):
            if not (task_dir / 'POSCAR').exists():
                continue
            task_path = str(task_dir.relative_to(self.work_dir))
            self.db.add_task(task_path, task_type)
            task_dirs.append(task_dir)
        return task_dirs

    def _check_registered_submit_tasks_completed(self, parent_dir: Path, task_type: str) -> bool:
        """检查目录下所有 task.* 是否已注册且成功。"""
        task_dirs = self._register_submit_tasks(parent_dir, task_type)
        if not task_dirs:
            return False

        for task_dir in task_dirs:
            task = self._get_db_task(task_dir)
            if not task or task['status'] != 'success':
                return False
        return True

    def _get_active_slurm_jobs(self, job_ids: Optional[Iterable[str]] = None) -> Optional[Set[str]]:
        """获取当前活跃的 SLURM job id；查询失败时返回 None。"""
        normalized_job_ids = None
        if job_ids is not None:
            normalized_job_ids = sorted({str(job_id).strip() for job_id in job_ids if str(job_id).strip()})
            if not normalized_job_ids:
                return set()

        command = ['squeue', '-h', '-o', '%i']
        query_scope = f"用户 {os.environ.get('USER', 'root')} 的作业"
        if normalized_job_ids is None:
            command.extend(['-u', os.environ.get('USER', 'root')])
        else:
            command.extend(['-j', ','.join(normalized_job_ids)])
            query_scope = f"{len(normalized_job_ids)} 个运行中任务"

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"查询 squeue 超时，跳过 running 状态校正: {query_scope}")
            return None
        except Exception as exc:
            logger.warning(f"查询 squeue 失败，跳过 running 状态校正: {query_scope} ({exc})")
            return None

        if result.returncode != 0:
            error = result.stderr.strip()
            if error:
                logger.warning(f"查询 squeue 失败，跳过 running 状态校正: {query_scope} ({error})")
            else:
                logger.warning(f"查询 squeue 失败，跳过 running 状态校正: {query_scope}")
            return None
        return {job_id for job_id in result.stdout.strip().split() if job_id}

    def _task_relpath(self, task_path: Path) -> str:
        """获取相对 work_dir 的任务路径。"""
        return str(task_path.relative_to(self.work_dir))

    def _get_db_task(self, task_path: Path) -> Optional[Dict]:
        """获取数据库中的任务记录。"""
        return self.db.get_task(self._task_relpath(task_path))

    def _get_tasks_under(self, path_prefix: Path, task_type: str = None,
                         status: str = None) -> List[Dict]:
        """获取某个目录前缀下的数据库任务。"""
        prefix = f"{path_prefix.relative_to(self.work_dir)}/"
        return self.db.get_tasks_by_prefix(prefix, task_type=task_type, status=status)

    def _generation_marker(self, struct_dir: Path, name: str) -> Path:
        """结构根目录下的任务生成标记。"""
        return struct_dir / f".generated__{name}"

    def _postprocess_marker(self, struct_dir: Path, name: str) -> Path:
        """结构根目录下的后处理标记。"""
        return struct_dir / f".postprocess__{name}"

    def _postprocess_failure_key(self, struct_dir: Path, stage: str,
                                 volume_name: Optional[str] = None) -> str:
        """当前 manager 进程内用于抑制重复后处理失败重试的 key。"""
        parts = [struct_dir.name, stage]
        if volume_name:
            parts.append(volume_name)
        return "::".join(parts)

    def _postprocess_error_file(self, struct_dir: Path, stage: str,
                                volume_name: Optional[str] = None) -> Path:
        """后处理错误日志文件。"""
        if volume_name is not None:
            error_dir = struct_dir / volume_name / 'analyze'
            error_dir.mkdir(exist_ok=True)
            return error_dir / 'postprocess_error.log'

        error_dir = struct_dir / 'analyze'
        error_dir.mkdir(exist_ok=True)
        return error_dir / f'{stage}_postprocess_error.log'

    def _record_postprocess_error(self, struct_dir: Path, stage: str, exc: Exception,
                                  volume_name: Optional[str] = None):
        """写入后处理失败日志，便于定位单个结构问题。"""
        error_file = self._postprocess_error_file(
            struct_dir,
            stage,
            volume_name=volume_name,
        )
        error_file.write_text(
            f"{datetime.now().isoformat()}\n"
            f"{type(exc).__name__}: {exc}\n"
        )

    def _clear_postprocess_error(self, struct_dir: Path, stage: str,
                                 volume_name: Optional[str] = None):
        """清理后处理失败日志。"""
        error_file = self._postprocess_error_file(
            struct_dir,
            stage,
            volume_name=volume_name,
        )
        if error_file.exists():
            error_file.unlink()

    def _workflow_has(self, structure_name: str, stage: str,
                      volume_name: Optional[str] = None,
                      pressure_name: Optional[str] = None,
                      state: Optional[str] = None) -> bool:
        """查询 workflow_state。"""
        return self.db.has_workflow_state(
            structure_name,
            stage,
            state=state,
            volume_name=volume_name,
            pressure_name=pressure_name,
        )

    def _workflow_set(self, structure_name: str, stage: str, state: str = 'done',
                      volume_name: Optional[str] = None,
                      pressure_name: Optional[str] = None,
                      source_task: Optional[str] = None):
        """写入 workflow_state。"""
        self.db.set_workflow_state(
            structure_name,
            stage,
            state=state,
            volume_name=volume_name,
            pressure_name=pressure_name,
            source_task=source_task,
        )

    def _workflow_clear(self, structure_name: str,
                        volume_name: Optional[str] = None,
                        pressure_name: Optional[str] = None,
                        stages: Optional[List[str]] = None):
        """删除 workflow_state。"""
        self.db.delete_workflow_states(
            structure_name=structure_name,
            volume_name=volume_name,
            pressure_name=pressure_name,
            stages=stages,
        )

    def _backfill_workflow_state_from_markers(self):
        """从已有 marker 文件回填 workflow_state，避免升级后重复后处理。"""
        if not self.structures_dir.exists():
            return

        for struct_dir in self.structures_dir.iterdir():
            if not struct_dir.is_dir():
                continue

            if (struct_dir / '.phonon_done').exists() or self._postprocess_marker(struct_dir, 'phonon__volume_1.0').exists():
                self._workflow_set(struct_dir.name, 'phonon_postprocessed', volume_name='volume_1.0')

            if self._postprocess_marker(struct_dir, 'qha').exists():
                self._workflow_set(struct_dir.name, 'qha_postprocessed')

            if (struct_dir / '.imaginary_frequency').exists() or (struct_dir / '.has_imag').exists():
                self._workflow_set(struct_dir.name, 'imaginary_checked', state='has_imaginary', volume_name='volume_1.0')

    def record_task_time(self, task_path: Path, start_time: str, end_time: str, duration: float, status: str):
        """记录任务执行时间

        Args:
            task_path: 任务路径（相对于work_dir）
            start_time: 开始时间 ISO格式
            end_time: 结束时间 ISO格式
            duration: 持续时间（秒）
            status: 任务状态（success/failed）
        """
        task_rel = str(task_path.relative_to(self.work_dir))
        task_hash = abs(hash(task_rel)) % 1000000
        record_file = self.task_times_dir / f"{task_hash:06d}.json"

        record = {
            'task_path': task_rel,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'status': status,
            'recorded_at': datetime.now().isoformat()
        }

        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)

    def sync_task_times(self):
        """同步任务执行时间（仅扫描数据库中已结束的已跟踪任务）"""
        for status in ('success', 'failed'):
            for task_data in self.db.get_tasks(status=status):
                task_dir = self.work_dir / task_data['path']
                if task_dir.exists():
                    self._extract_task_time(task_dir, status)

    def _extract_task_time(self, task_dir: Path, task_status: str):
        """从任务目录的.task_time文件提取执行时间并更新到队列"""
        if task_status not in ['success', 'failed']:
            return

        # 获取任务相对路径
        task_rel = str(task_dir.relative_to(self.work_dir))

        # 检查是否已记录过（避免重复记录）
        task_hash = abs(hash(task_rel)) % 1000000
        record_file = self.task_times_dir / f"{task_hash:06d}.json"
        if record_file.exists():
            return  # 已记录过

        # 优先读取 .task_time 文件
        task_time_file = task_dir / '.task_time'
        if task_time_file.exists():
            content = task_time_file.read_text()
            time_data = {}
            for line in content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    time_data[key.strip()] = value.strip()

            start_time = time_data.get('start_time', '')
            end_time = time_data.get('end_time', '')
            duration = float(time_data.get('duration_seconds', 0))
            task_status = time_data.get('status', task_status)
            if start_time and end_time:
                self.record_task_time(task_dir, start_time, end_time, duration, task_status)
                self.db.update_task_time(task_rel, start_time, end_time, duration, task_status)
                return

        # 回退：从SLURM日志文件中解析时间
        slurm_logs = list(task_dir.glob('slurm_*.log'))
        if not slurm_logs:
            return

        log_file = max(slurm_logs, key=lambda p: p.stat().st_mtime)
        with open(log_file, 'r') as f:
            lines = f.readlines()

        start_time = None
        end_time = datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()

        for line in lines:
            if 'Date:' in line:
                date_str = line.split('Date:')[-1].strip()
                start_time = datetime.strptime(date_str, '%a %b %d %I:%M:%S %p %Z %Y').isoformat()
                break

        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration = (end_dt - start_dt).total_seconds()
            self.record_task_time(task_dir, start_time, end_time, duration, task_status)
            self.db.update_task_time(task_rel, start_time, end_time, duration, task_status)

    # ========== 队列同步 ==========

    def sync_queue_from_filesystem(self):
        """完整的队列同步：扫描文件系统并更新队列状态"""
        logger.info("=== 开始队列同步 ===")

        synced_counts = {
            'added': 0,
            'updated_success': 0,
            'updated_failed': 0,
            'updated_running': 0,
            'removed': 0
        }

        # 1. 扫描所有任务目录，记录存在的任务路径
        all_task_paths = set()

        if self.structures_dir.exists():
            struct_dirs = list(self.structures_dir.iterdir())
            logger.info(f"正在扫描 {len(struct_dirs)} 个结构目录...")
            for i, struct_dir in enumerate(struct_dirs):
                if not struct_dir.is_dir():
                    continue

                if (i + 1) % 20 == 0:
                    logger.info(f"  已扫描 {i + 1}/{len(struct_dirs)} 个结构目录")

                # 扫描opt任务
                opt_dir = struct_dir / 'opt'
                if opt_dir.exists() and (opt_dir / 'POSCAR').exists():
                    task_path = str(opt_dir.relative_to(self.work_dir))
                    all_task_paths.add(task_path)

                    # 无论任务是否已存在，都要用文件系统状态回写数据库。
                    if self.db.add_task(task_path, 'opt'):
                        synced_counts['added'] += 1

                    status = self.get_task_status(opt_dir)
                    if status == 'success':
                        self.db.update_status(task_path, status)
                        synced_counts['updated_success'] += 1

                # 扫描phonon/qha任务
                for volume_dir in struct_dir.glob('volume_*'):
                    for task_dir in self._iter_submit_task_dirs(volume_dir):
                        if not task_dir.is_dir():
                            continue
                        if not (task_dir / 'POSCAR').exists():
                            continue

                        task_path = str(task_dir.relative_to(self.work_dir))
                        all_task_paths.add(task_path)

                        # 判断任务类型
                        task_type = 'qha' if volume_dir.name != 'volume_1.0' else 'phonon'

                        if self.db.add_task(task_path, task_type):
                            synced_counts['added'] += 1

                        status = self.get_task_status(task_dir)
                        if status == 'success':
                            self.db.update_status(task_path, status)
                            synced_counts['updated_success'] += 1

                # 扫描BTE任务 (P_XXGPa/opt, P_XXGPa/bte/fc2/task.*, P_XXGPa/bte/fc3/task.*)
                for p_dir in struct_dir.glob('P_*GPa'):
                    if not p_dir.is_dir():
                        continue

                    # 压强点 opt 任务
                    p_opt_dir = p_dir / 'opt'
                    if p_opt_dir.exists() and (p_opt_dir / 'POSCAR').exists():
                        task_path = str(p_opt_dir.relative_to(self.work_dir))
                        all_task_paths.add(task_path)

                        if self.db.add_task(task_path, 'bte_opt'):
                            synced_counts['added'] += 1

                        status = self.get_task_status(p_opt_dir)
                        if status == 'success':
                            self.db.update_status(task_path, status)
                            synced_counts['updated_success'] += 1

                    # BTE fc2/fc3 任务
                    bte_dir = p_dir / 'bte'
                    if bte_dir.exists():
                        for fc_type in ['fc2', 'fc3']:
                            fc_dir = bte_dir / fc_type
                            if not fc_dir.exists():
                                continue
                            bte_task_type = f'bte_{fc_type}'
                            for task_dir in self._iter_submit_task_dirs(fc_dir):
                                if not task_dir.is_dir():
                                    continue
                                if not (task_dir / 'POSCAR').exists():
                                    continue

                                task_path = str(task_dir.relative_to(self.work_dir))
                                all_task_paths.add(task_path)

                                if self.db.add_task(task_path, bte_task_type):
                                    synced_counts['added'] += 1

                                status = self.get_task_status(task_dir)
                                if status == 'success':
                                    self.db.update_status(task_path, status)
                                    synced_counts['updated_success'] += 1

        # 2. 删除队列中不存在的任务
        removed = self.db.remove_nonexistent_tasks(all_task_paths)
        synced_counts['removed'] = removed

        # 3. 输出同步结果
        logger.info(f"队列同步完成: 新增={synced_counts['added']}, "
                   f"success={synced_counts['updated_success']}, "
                   f"failed={synced_counts['updated_failed']}, "
                   f"running={synced_counts['updated_running']}, "
                   f"删除={synced_counts['removed']}")

        return synced_counts

    def _scan_submit_candidates(self, plain_only: bool) -> List[Dict]:
        """快速扫描可提交目录。"""
        return self.submit_scanner.scan(plain_only=plain_only)

    def _sync_submit_candidates(self, plain_only: bool, scan_name: str):
        """递归扫描可提交目录并同步数据库。"""
        logger.info(f"=== {scan_name} ===")

        records = self._scan_submit_candidates(plain_only)
        registered_counts = self.db.add_tasks_ignore_existing(records)
        synced_counts = {
            'added': registered_counts['added'],
            'updated_success': 0,
            'updated_failed': 0,
            'updated_running': 0,
            'removed': 0,
        }

        all_task_paths = {record['path'] for record in records}
        for record in records:
            task_path = record['path']
            task_dir = self.work_dir / task_path
            status = self.get_task_status(task_dir)
            if status == 'success':
                self.db.update_status(task_path, status)
                synced_counts['updated_success'] += 1

        removed = 0
        for task_data in self.db.get_tasks():
            task_path = task_data['path']
            if self.submit_scanner.is_submit_candidate_name(Path(task_path).name, plain_only) and task_path not in all_task_paths:
                if self.db.remove_task(task_path):
                    removed += 1
        synced_counts['removed'] = removed
        logger.info(
            f"{scan_name} 完成: "
            f"新增={synced_counts['added']}, "
            f"success={synced_counts['updated_success']}, "
            f"failed={synced_counts['updated_failed']}, "
            f"running={synced_counts['updated_running']}, "
            f"删除={synced_counts['removed']}"
        )
        return synced_counts

    def _register_submit_candidates(self, plain_only: bool, scan_name: str):
        """快速扫描可提交目录，只注册新增任务。"""
        logger.info(f"=== {scan_name} ===")

        registered_counts = self.db.add_tasks_ignore_existing(
            self._scan_submit_candidates(plain_only)
        )

        logger.info(
            f"{scan_name} 完成: "
            f"新增={registered_counts['added']}, "
            f"已存在={registered_counts['existing']}"
        )
        return registered_counts

    def sync_plain_submit_tasks(self):
        """plain_submit 模式：递归扫描所有 task.* 目录并同步数据库。"""
        return self._sync_submit_candidates(
            True,
            'plain_submit 扫描 task.* 目录'
        )

    def sync_all_submit_tasks(self):
        """普通 sync：递归扫描 opt/task.* 并同步数据库。"""
        return self._sync_submit_candidates(
            False,
            '递归扫描 opt/task.* 目录'
        )

    def register_plain_submit_tasks(self):
        """plain_submit 模式：只注册所有 task.* 目录。"""
        return self._register_submit_candidates(
            True,
            'plain_submit 注册 task.* 目录'
        )

    def register_all_submit_tasks(self):
        """普通 sync：只注册 opt/task.*。"""
        return self._register_submit_candidates(
            False,
            '递归注册 opt/task.* 目录'
        )

    def reconcile_tracked_tasks(self):
        """启动恢复：仅修正已跟踪任务中的缺失目录和 success 标记。"""
        logger.info("=== 恢复已跟踪任务状态 ===")

        running_recovered = self.reconcile_tracked_running_tasks()

        recovered = {
            'success': 0,
            'removed': 0,
        }

        for task_data in self.db.get_tasks():
            task_path_str = task_data['path']
            task_dir = self.work_dir / task_path_str

            if task_data['status'] == 'running':
                continue

            if not task_dir.exists() or not (task_dir / 'POSCAR').exists():
                if self.db.remove_task(task_path_str):
                    recovered['removed'] += 1
                continue

            if task_data['status'] != 'success' and (task_dir / '.success').exists():
                self.db.update_status(task_path_str, 'success')
                recovered['success'] += 1

        logger.info(
            "已恢复跟踪任务状态: "
            f"running_success={running_recovered['success']}, "
            f"running_failed={running_recovered['failed']}, "
            f"running_pending={running_recovered['pending']}, "
            f"running_removed={running_recovered['removed']}, "
            f"success={recovered['success']}, "
            f"removed={recovered['removed']}"
        )
        return recovered

    def reconcile_tracked_running_tasks(self):
        """plain_submit 启动恢复：只处理数据库中 running 的任务。"""
        logger.info("=== 恢复运行中任务状态 ===")

        recovered = {
            'success': 0,
            'failed': 0,
            'pending': 0,
            'removed': 0,
        }
        running_tasks = self.db.get_running_tasks()
        running_job_ids = [task_data.get('slurm_job_id') for task_data in running_tasks if task_data.get('slurm_job_id')]
        active_jobs = self._get_active_slurm_jobs(running_job_ids)
        status_updates = []
        pending_paths = []
        stale_job_ids = []

        for task_data in running_tasks:
            task_path_str = task_data['path']
            task_dir = self.work_dir / task_path_str
            slurm_job_id = task_data.get('slurm_job_id')

            if not task_dir.exists() or not (task_dir / 'POSCAR').exists():
                status_updates.append((task_path_str, 'failed'))
                stale_job_ids.append(slurm_job_id)
                recovered['failed'] += 1
                recovered['removed'] += 1
                continue

            if (task_dir / '.success').exists():
                clear_task_status(task_dir, self.config, statuses=['running', 'failed'])
                status_updates.append((task_path_str, 'success'))
                stale_job_ids.append(slurm_job_id)
                recovered['success'] += 1
            elif not slurm_job_id:
                pending_paths.append(task_path_str)
                stale_job_ids.append(slurm_job_id)
                recovered['pending'] += 1
            elif active_jobs is not None and slurm_job_id not in active_jobs:
                clear_task_status(task_dir, self.config, statuses=['running', 'failed'])
                status_updates.append((task_path_str, 'failed'))
                stale_job_ids.append(slurm_job_id)
                recovered['failed'] += 1

        self.db.reset_tasks_to_pending_bulk(pending_paths)
        self.db.update_status_bulk(status_updates)
        self._remove_job_mappings(stale_job_ids)

        logger.info(
            "已恢复运行中任务状态: "
            f"success={recovered['success']}, "
            f"failed={recovered['failed']}, "
            f"pending={recovered['pending']}, "
            f"removed={recovered['removed']}"
        )
        return recovered

    def sync_running_tasks_status(self):
        """快速同步：只检查running任务的文件系统状态"""
        running_tasks = self.db.get_running_tasks()
        if not running_tasks:
            return

        synced = 0
        reset_to_failed = 0
        reset_to_pending = 0
        running_job_ids = [task_data.get('slurm_job_id') for task_data in running_tasks if task_data.get('slurm_job_id')]
        active_jobs = self._get_active_slurm_jobs(running_job_ids)
        status_updates = []
        pending_paths = []
        stale_job_ids = []

        for task_data in running_tasks:
            task_path_str = task_data['path']
            task_path = self.work_dir / task_path_str
            slurm_job_id = task_data.get('slurm_job_id', '')

            if not task_path.exists() or not (task_path / 'POSCAR').exists():
                status_updates.append((task_path_str, 'failed'))
                stale_job_ids.append(slurm_job_id)
                reset_to_failed += 1
                continue

            if (task_path / '.success').exists():
                clear_task_status(task_path, self.config, statuses=['running', 'failed'])
                status_updates.append((task_path_str, 'success'))
                stale_job_ids.append(slurm_job_id)
                synced += 1
            elif not slurm_job_id:
                pending_paths.append(task_path_str)
                stale_job_ids.append(slurm_job_id)
                reset_to_pending += 1
                logger.warning(f"任务 {task_path_str} 缺少 slurm_job_id，重置为pending")
            elif active_jobs is not None and slurm_job_id not in active_jobs:
                clear_task_status(task_path, self.config, statuses=['running', 'failed'])
                status_updates.append((task_path_str, 'failed'))
                stale_job_ids.append(slurm_job_id)
                reset_to_failed += 1
                logger.warning(f"任务 {task_path_str} SLURM job {slurm_job_id} 已消失，标记为failed")

        self.db.reset_tasks_to_pending_bulk(pending_paths)
        self.db.update_status_bulk(status_updates)
        self._remove_job_mappings(stale_job_ids)

        if synced > 0:
            logger.info(f"同步了 {synced} 个running任务状态")
        if reset_to_failed > 0:
            logger.info(f"标记了 {reset_to_failed} 个异常终止任务为failed")
        if reset_to_pending > 0:
            logger.info(f"重置了 {reset_to_pending} 个异常任务为pending")

    # ========== Sbatch任务提交 ==========

    def _generate_task_name(self, task_dir: Path) -> str:
        """生成任务名称用于sbatch job-name

        例如: opt_mp-1234, phonon_mp-1234_vol1.0_task001, bte_mp-1234_fc2_t001
        """
        parts = task_dir.parts

        # 查找mp-xxx部分
        mp_id = None
        for part in parts:
            if part.startswith('mp-'):
                mp_id = part
                break

        if 'opt' in parts:
            return f"opt_{mp_id}" if mp_id else "opt_task"

        if 'task.BTE' in parts:
            pressure = None
            for part in parts:
                if re.match(r'P_\d+GPa', part):
                    pressure = part
                    break
            if mp_id and pressure:
                return f"btepost_{mp_id}_{pressure}"
            return "btepost_task"

        # BTE任务
        if 'bte' in parts:
            fc_type = None
            task_num = None
            for part in parts:
                if part in ('fc2', 'fc3'):
                    fc_type = part
                if part.startswith('task.'):
                    task_num = part.replace('task.', 't')
            if mp_id and fc_type and task_num:
                return f"bte_{mp_id}_{fc_type}_{task_num}"
            return "bte_task"

        # phonon/qha任务
        volume = None
        task_num = None
        for part in parts:
            if part.startswith('volume_'):
                volume = part.replace('volume_', 'v')
            if part.startswith('task.'):
                task_num = part.replace('task.', 't')

        if mp_id and volume and task_num:
            return f"ph_{mp_id}_{volume}_{task_num}"

        return "qflow_task"

    def submit_sbatch_task(self, task_dir: Path, task_type: str = 'opt') -> Optional[str]:
        """为任务生成sbatch脚本并提交

        Returns: job_id或None（失败时）
        """
        # 生成任务名称
        task_name = self._generate_task_name(task_dir)

        # 生成sbatch脚本
        script_content = generate_task_script(self.config, task_name, task_type=task_type)

        # 写入脚本文件到任务目录
        script_file = task_dir / 'run.sbatch'
        script_file.write_text(script_content)
        script_file.chmod(0o755)

        # 提交前清理旧兼容标记和错误日志，success 标记保留为跳过依据
        clear_task_status(task_dir, self.config, remove_error_log=True)

        # 提交sbatch
        result = subprocess.run(
            ['sbatch', 'run.sbatch'],
            cwd=task_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # 解析job_id: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"  提交任务: {task_dir.name} -> job {job_id}")

            # 保存job_id映射
            self._save_job_mapping(job_id, task_dir)

            return job_id
        else:
            logger.error(f"  提交失败: {task_dir.name}\n{result.stderr}")
            return None

    def _save_job_mapping(self, job_id: str, task_path: Path):
        """保存job_id到任务路径的映射"""
        jobs = {}
        if self.jobs_file.exists():
            jobs = json.loads(self.jobs_file.read_text())

        jobs[job_id] = str(task_path.relative_to(self.work_dir))
        self.jobs_file.write_text(json.dumps(jobs, indent=2))

    def _remove_job_mapping(self, job_id: str):
        """移除已结束或失效的 job_id 映射。"""
        if not job_id or not self.jobs_file.exists():
            return

        jobs = json.loads(self.jobs_file.read_text())

        if job_id in jobs:
            del jobs[job_id]
            self.jobs_file.write_text(json.dumps(jobs, indent=2))

    def _remove_job_mappings(self, job_ids: List[str]):
        """批量移除已结束或失效的 job_id 映射。"""
        if not job_ids or not self.jobs_file.exists():
            return

        stale_ids = {job_id for job_id in job_ids if job_id}
        if not stale_ids:
            return

        jobs = json.loads(self.jobs_file.read_text())
        changed = False
        for job_id in stale_ids:
            if job_id in jobs:
                del jobs[job_id]
                changed = True

        if changed:
            self.jobs_file.write_text(json.dumps(jobs, indent=2))

    def submit_pending_tasks(self):
        """从队列获取pending任务并提交（受max_workers限制）"""
        # 读取最大并发任务数，默认为1
        max_workers = 1  # 默认值
        if self.max_workers_file.exists():
            max_workers = int(self.max_workers_file.read_text().strip())

        logger.info(f"最大并发任务数: {max_workers}")

        # 从队列获取running任务数
        running_tasks = self.db.get_tasks(status='running')
        current_running = len(running_tasks)

        logger.info(f"当前运行任务数: {current_running}")

        # 计算还能提交多少任务
        slots_available = max_workers - current_running
        if slots_available <= 0:
            logger.info(f"已达到最大并发数限制 ({max_workers})，跳过提交")
            return
        logger.info(f"还可提交 {slots_available} 个任务")

        submitted = 0

        # 从队列获取pending任务并提交
        while submitted < slots_available:
            task_data = self.db.get_pending_task()
            if task_data is None:
                break  # 没有pending任务了

            task_path_str = task_data['path']
            task_dir = self.work_dir / task_path_str

            requires_poscar = task_data.get('task_type') != 'bte_postprocess'
            if not task_dir.exists() or (requires_poscar and not (task_dir / 'POSCAR').exists()):
                # 任务目录不存在，标记为failed
                self.db.update_status(task_path_str, 'failed')
                logger.warning(f"  任务目录不存在，标记为failed: {task_path_str}")
                continue

            marker_status = self.get_task_status(task_dir)
            if marker_status == 'success':
                clear_task_status(task_dir, self.config, statuses=['running', 'failed'])
                self.db.update_status(task_path_str, 'success')
                logger.info(f"  跳过已完成任务: {task_path_str}")
                continue

            # 提交前实时生成VASP输入文件（INCAR/POTCAR/KPOINTS）
            task_type = task_data.get('task_type', 'opt')
            vasp_type_map = {
                'opt': 'opt', 'qha_opt': 'qha_opt', 'phonon': 'phonon',
                'bte_opt': 'bte_opt', 'bte_fc2': 'phonon', 'bte_fc3': 'phonon',
                'qha': 'phonon', 'plain': 'plain',
            }
            vasp_type = vasp_type_map.get(task_type, 'phonon')
            if self.worker_mode == 'vasp' and task_type != 'bte_postprocess':
                self._generate_vasp_inputs(task_dir, task_type=vasp_type)

            # 提交sbatch任务
            job_id = self.submit_sbatch_task(task_dir, task_type=task_type)
            if job_id:
                self.db.update_status(task_path_str, 'running', slurm_job_id=job_id)
                submitted += 1
            else:
                # 提交失败，标记为failed
                self.db.update_status(task_path_str, 'failed')

        if submitted > 0:
            logger.info(f"提交了 {submitted} 个新任务 (当前运行: {current_running + submitted}/{max_workers})")

    def sync_running_tasks(self):
        """同步running任务状态（运行中定期调用的快速版本）"""
        self.sync_running_tasks_status()

    # ========== 统计信息 ==========

    def collect_statistics(self) -> Dict[str, Dict[str, int]]:
        """收集任务统计信息（从队列获取）"""
        return self.db.get_statistics()

    # ========== 原有的任务生成和后处理方法 ==========

    def get_all_structures(self) -> List[Path]:
        """获取所有结构目录"""
        if not self.structures_dir.exists():
            return []
        return [d for d in self.structures_dir.iterdir() if d.is_dir()]

    def _build_qha_scan_state(self) -> Dict[str, Set]:
        """批量加载 QHA 工作流常用状态，避免按结构反复查库。"""
        opt_success = {
            task['structure_name']
            for task in self.db.get_tasks(task_type='opt', status='success')
            if task.get('structure_name')
        }

        phonon_registered = {
            (row['structure_name'], row['volume_name'])
            for row in self.db.get_workflow_states(stage='phonon_generated')
            if row.get('structure_name')
        }

        qha_opt_registered = {
            (row['structure_name'], row['volume_name'])
            for row in self.db.get_workflow_states(stage='qha_opt_generated')
            if row.get('structure_name')
        }

        qha_opt_success = {
            (task['structure_name'], task['volume_name'])
            for task in self.db.get_tasks(task_type='qha_opt', status='success')
            if task.get('structure_name') and task.get('volume_name')
        }

        qha_registered = {
            (row['structure_name'], row['volume_name'])
            for row in self.db.get_workflow_states(stage='qha_generated')
            if row.get('structure_name')
        }

        return {
            'opt_success': opt_success,
            'phonon_registered': phonon_registered,
            'qha_opt_registered': qha_opt_registered,
            'qha_opt_success': qha_opt_success,
            'qha_registered': qha_registered,
        }

    def _refine_structure(self, poscar_path: Path) -> None:
        """使用pymatgen标准化结构（保持primitive cell）"""
        try:
            struct = PMGStructure.from_file(str(poscar_path))
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            # 使用primitive cell而不是conventional cell，减少原子数
            refined = sga.get_primitive_standard_structure()
            refined.to(filename=str(poscar_path), fmt="poscar")
            logger.info(f"    已标准化结构: {poscar_path.parent.name} ({len(refined)} atoms)")
        except Exception as e:
            logger.warning(f"    结构标准化失败 {poscar_path}: {e}")

    def generate_opt_tasks(self, struct_dirs: Optional[List[Path]] = None):
        """生成结构优化任务"""
        logger.info("=== 扫描结构优化任务 ===")

        if struct_dirs is None:
            struct_dirs = self.get_all_structures()

        for struct_dir in struct_dirs:
            opt_dir = struct_dir / 'opt'

            # 如果opt目录不存在但结构目录下有POSCAR，自动创建opt任务
            if not opt_dir.exists():
                poscar = struct_dir / 'POSCAR'
                if poscar.exists():
                    logger.info(f"  创建opt任务: {struct_dir.name}")
                    opt_dir.mkdir(exist_ok=True)
                    shutil.copy(str(poscar), str(opt_dir / 'POSCAR'))
                    if self.refine_structure:
                        self._refine_structure(opt_dir / 'POSCAR')
                    # 添加到任务队列（VASP输入文件在提交时实时生成）
                    task_path = str(opt_dir.relative_to(self.work_dir))
                    self.db.add_task(task_path, 'opt')
                    self._workflow_set(struct_dir.name, 'opt_generated', source_task=task_path)

    def check_opt_completed(self, struct_dir: Path) -> bool:
        """检查结构优化是否完成"""
        opt_dir = struct_dir / 'opt'
        if not opt_dir.exists():
            return False
        task = self._get_db_task(opt_dir)
        return bool(task and task['status'] == 'success')

    def _get_optimized_structure(self, struct_dir: Path):
        """获取优化后的结构"""
        opt_dir = struct_dir / 'opt'
        contcar = opt_dir / 'CONTCAR'
        if contcar.exists() and contcar.stat().st_size > 0:
            return read(str(contcar))
        poscar = opt_dir / 'POSCAR'
        if poscar.exists():
            return read(str(poscar))
        return None

    def generate_phonon_tasks(self, struct_dirs: Optional[List[Path]] = None,
                              scan_state: Optional[Dict[str, Set]] = None):
        """生成声子计算任务"""
        logger.info("=== 生成声子计算任务 ===")

        if struct_dirs is None:
            struct_dirs = self.get_all_structures()
        if scan_state is None:
            scan_state = self._build_qha_scan_state()

        opt_success = scan_state['opt_success']
        phonon_registered = scan_state['phonon_registered']

        for struct_dir in struct_dirs:
            volume_dir = struct_dir / 'volume_1.0'
            volume_key = (struct_dir.name, volume_dir.name)
            if volume_key in phonon_registered and not volume_dir.exists():
                self._workflow_clear(struct_dir.name, volume_name=volume_dir.name, stages=['phonon_generated'])
                phonon_registered.discard(volume_key)

            if struct_dir.name not in opt_success:
                continue

            if volume_key in phonon_registered:
                continue

            if not volume_dir.exists():
                self._prepare_phonon_tasks(struct_dir, volume_dir)
                phonon_registered.add(volume_key)
                continue

            if volume_dir.exists():
                task_dirs = self._register_submit_tasks(volume_dir, 'phonon')

                if task_dirs:
                    phonon_registered.add(volume_key)
                    self._workflow_set(struct_dir.name, 'phonon_generated', volume_name=volume_dir.name)

    def _prepare_phonon_tasks(self, struct_dir: Path, volume_dir: Path):
        """准备声子计算任务"""
        logger.info(f"  生成声子任务: {struct_dir.name}")

        atoms = self._get_optimized_structure(struct_dir)
        if atoms is None:
            logger.warning(f"  无法读取结构 {struct_dir.name}")
            return

        poscar_file = struct_dir / 'POSCAR'
        if not poscar_file.exists():
            write(str(poscar_file), atoms, format='vasp', vasp5=True)

        supercell = self.phonon_config.get('supercell', None)
        max_atoms = self.phonon_config.get('max_atoms', None)
        min_atoms = self.phonon_config.get('min_atoms', 100)
        min_length = self.phonon_config.get('min_length', 10.0)
        distance = self.phonon_config.get('displacement_distance', 0.01)

        n_tasks = generate_phonon_displacements(
            atoms=atoms,
            volume_dir=str(volume_dir),
            supercell=supercell,
            max_atoms=max_atoms,
            min_atoms=min_atoms,
            min_length=min_length,
            distance=distance
        )
        logger.info(f"    生成 {n_tasks} 个位移任务")

        # 添加phonon任务到队列，包含 task.perfect
        self._register_submit_tasks(volume_dir, 'phonon')

        self._workflow_set(struct_dir.name, 'phonon_generated', volume_name=volume_dir.name)

        # 声子位移任务的VASP输入文件在提交时实时生成

    def _generate_vasp_inputs(self, task_dir: Path, task_type: str = 'opt'):
        """生成VASP输入文件

        每次调用时实时读取 config.yaml 的 incar 设置，确保使用最新配置。

        Args:
            task_dir: 任务目录
            task_type: 任务类型
                - 'opt': 结构优化，使用 MPRelaxSet + ISIF=3
                - 'qha_opt': QHA体积点优化，使用 MPRelaxSet + ISIF=2
                - 'bte_opt': BTE压强点优化，使用 MPRelaxSet + ISIF=3 + PSTRESS
                - 'phonon': 声子/BTE单点计算，默认使用 MatPESStaticSet
                - 'plain': plain_submit 任务，默认使用 MatPESStaticSet
        """
        import warnings
        import re
        warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='pymatgen')

        poscar = task_dir / 'POSCAR'
        if not poscar.exists():
            return

        # 实时读取最新的 config.yaml incar 设置
        fresh_config = load_config()
        incar_config = fresh_config.get('incar', {})
        vasp_set_config = fresh_config.get('vasp_sets', {})
        incar_opt = incar_config.get('opt', {})
        incar_phonon = incar_config.get('phonon', {})
        incar_plain = incar_config.get('plain', {})
        incar_qha_opt = incar_config.get('qha_opt', {})

        # 兼容旧配置格式
        if not incar_opt and not incar_phonon:
            incar_opt = incar_config
            incar_phonon = incar_config

        if not incar_plain:
            incar_plain = incar_phonon or incar_opt

        # 如果没有配置qha_opt，使用opt的配置但修改ISIF=2
        if not incar_qha_opt and incar_opt:
            incar_qha_opt = incar_opt.copy()
            incar_qha_opt['ISIF'] = 2

        structure = PMGStructure.from_file(str(poscar))

        def resolve_vasp_set_name(current_task_type: str) -> str:
            configured = vasp_set_config.get(current_task_type)
            if configured:
                return str(configured).strip().lower()

            default_map = {
                'opt': 'mprelax',
                'qha_opt': 'mprelax',
                'bte_opt': 'mprelax',
                'phonon': 'matpes',
                'plain': 'matpes',
            }
            return default_map.get(current_task_type, 'mprelax')

        def build_vasp_set(set_name: str, incar_settings: dict):
            alias_map = {
                'matpes': 'matpes',
                'matpesstatic': 'matpes',
                'matpesstaticset': 'matpes',
                'mpstatic': 'mpstatic',
                'mpstaticset': 'mpstatic',
                'mprelax': 'mprelax',
                'mprelaxed': 'mprelax',
                'mprelaxset': 'mprelax',
            }
            normalized = alias_map.get(set_name)
            if normalized is None:
                raise ValueError(f"Unsupported vasp set '{set_name}' for task {task_type}")

            if normalized == 'matpes':
                return MatPESStaticSet(
                    structure,
                    user_incar_settings=incar_settings,
                    user_potcar_functional=self.potcar_functional,
                )
            if normalized == 'mpstatic':
                return MPStaticSet(
                    structure,
                    user_incar_settings=incar_settings,
                    user_potcar_functional=self.potcar_functional,
                )
            return MPRelaxSet(
                structure,
                user_incar_settings=incar_settings,
                user_potcar_functional=self.potcar_functional,
            )

        if task_type == 'phonon':
            incar_settings = incar_phonon
        elif task_type == 'plain':
            incar_settings = incar_plain
        elif task_type == 'qha_opt':
            incar_settings = incar_qha_opt
        elif task_type == 'bte_opt':
            # BTE 压强优化: ISIF=3 + PSTRESS
            incar_settings = incar_opt.copy()
            # 从目录名提取压强: P_XXGPa/opt -> XX GPa -> PSTRESS = XX * 10 (kBar)
            pressure_gpa = 0
            for part in task_dir.parts:
                m = re.match(r'P_(\d+)GPa', part)
                if m:
                    pressure_gpa = int(m.group(1))
                    break
            incar_settings['PSTRESS'] = pressure_gpa * 10  # GPa -> kBar
        else:
            incar_settings = incar_opt

        set_name = resolve_vasp_set_name(task_type)
        mp_set = build_vasp_set(set_name, incar_settings)

        mp_set.write_input(str(task_dir))

        # 如果设置了 METAGGA，必须删掉 GGA，否则 VASP 报错
        incar_path = task_dir / 'INCAR'
        if incar_path.exists() and 'METAGGA' in (incar_settings or {}):
            lines = incar_path.read_text().splitlines(keepends=True)
            lines = [l for l in lines if not l.strip().startswith('GGA')]
            incar_path.write_text(''.join(lines))

    def check_phonon_completed(self, struct_dir: Path) -> bool:
        """检查所有声子任务是否完成"""
        volume_dir = struct_dir / 'volume_1.0'
        if not volume_dir.exists():
            return False
        return self._check_registered_submit_tasks_completed(volume_dir, 'phonon')

    def check_imaginary_frequency_wrapper(self, struct_dir: Path) -> bool:
        """检查是否存在虚频"""
        if (struct_dir / '.imaginary_frequency').exists():
            return True
        if self._workflow_has(struct_dir.name, 'imaginary_checked', state='has_imaginary', volume_name='volume_1.0'):
            return True
        if self._workflow_has(struct_dir.name, 'imaginary_checked', state='clear', volume_name='volume_1.0'):
            return False

        volume_dir = struct_dir / 'volume_1.0'
        if not volume_dir.exists():
            return False

        cached = self.imaginary_db.get_cached_result(volume_dir)
        if cached is not None:
            self._workflow_set(
                struct_dir.name,
                'imaginary_checked',
                state='has_imaginary' if cached else 'clear',
                volume_name=volume_dir.name,
            )
            return cached

        has_imaginary = check_imaginary_frequency(str(volume_dir))
        self.imaginary_db.set_cached_result(volume_dir, has_imaginary)
        self._workflow_set(
            struct_dir.name,
            'imaginary_checked',
            state='has_imaginary' if has_imaginary else 'clear',
            volume_name=volume_dir.name,
        )
        return has_imaginary

    def generate_qha_tasks(self, struct_dirs: Optional[List[Path]] = None,
                           scan_state: Optional[Dict[str, Set]] = None):
        """生成QHA任务"""
        logger.info("=== 生成QHA计算任务 ===")

        if struct_dirs is None:
            struct_dirs = self.get_all_structures()
        if scan_state is None:
            scan_state = self._build_qha_scan_state()

        qha_opt_registered = scan_state['qha_opt_registered']

        for struct_dir in struct_dirs:
            qha_opt_targets = [
                (struct_dir.name, f'volume_{vol}')
                for vol in self.default_volumes if vol != 1.0
            ]
            if qha_opt_targets and all(target in qha_opt_registered for target in qha_opt_targets):
                continue

            volume_1_analyze = struct_dir / 'volume_1.0' / 'analyze' / 'phonopy_params.yaml'
            if not volume_1_analyze.exists():
                continue

            if (struct_dir / '.imaginary_frequency').exists():
                continue

            if self.check_imaginary_frequency_wrapper(struct_dir):
                logger.info(f"跳过 {struct_dir.name}：存在虚频")
                imaginary_flag = struct_dir / '.imaginary_frequency'
                imaginary_flag.touch()
                continue

            self._prepare_qha_volumes(struct_dir, qha_opt_registered)

    def _prepare_qha_volumes(self, struct_dir: Path,
                             qha_opt_registered: Optional[Set[Tuple[str, str]]] = None):
        """准备QHA所需的各体积点 - 第一步：创建优化任务"""
        atoms = self._get_optimized_structure(struct_dir)
        if atoms is None:
            return

        if qha_opt_registered is None:
            qha_opt_registered = {
                (task['structure_name'], task['volume_name'])
                for task in self.db.get_tasks(task_type='qha_opt')
                if task.get('structure_name') and task.get('volume_name')
            }

        for vol in self.default_volumes:
            if vol == 1.0:
                continue

            volume_dir = struct_dir / f'volume_{vol}'
            volume_key = (struct_dir.name, volume_dir.name)
            opt_dir = volume_dir / 'opt'
            if volume_key in qha_opt_registered and not opt_dir.exists():
                self._workflow_clear(struct_dir.name, volume_name=volume_dir.name, stages=['qha_opt_generated'])
                qha_opt_registered.discard(volume_key)
            elif volume_key in qha_opt_registered:
                continue

            # 如果优化目录已存在，跳过
            if opt_dir.exists():
                # 确保优化任务已注册到数据库
                task_path = str(opt_dir.relative_to(self.work_dir))
                self.db.add_task(task_path, 'qha_opt')
                qha_opt_registered.add(volume_key)
                self._workflow_set(struct_dir.name, 'qha_opt_generated', volume_name=volume_dir.name, source_task=task_path)
                continue

            logger.info(f"  创建体积 {vol} 的优化任务: {struct_dir.name}")

            # 缩放晶胞
            atoms_scaled = atoms.copy()
            scale_factor = vol ** (1/3)
            atoms_scaled.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)

            # 创建优化任务目录
            opt_dir.mkdir(parents=True, exist_ok=True)

            # 写入POSCAR
            poscar_path = opt_dir / 'POSCAR'
            write(str(poscar_path), atoms_scaled, format='vasp')

            # VASP输入文件在提交时实时生成

            # 添加优化任务到数据库
            task_path = str(opt_dir.relative_to(self.work_dir))
            self.db.add_task(task_path, 'qha_opt')
            qha_opt_registered.add(volume_key)
            self._workflow_set(struct_dir.name, 'qha_opt_generated', volume_name=volume_dir.name, source_task=task_path)
            logger.info(f"    已创建优化任务: {task_path}")

    def generate_qha_phonon_tasks(self, struct_dirs: Optional[List[Path]] = None,
                                  scan_state: Optional[Dict[str, Set]] = None):
        """生成QHA声子任务 - 在优化完成后"""
        logger.info("=== 检查QHA优化并生成声子任务 ===")

        if struct_dirs is None:
            struct_dirs = self.get_all_structures()
        if scan_state is None:
            scan_state = self._build_qha_scan_state()

        qha_opt_success = scan_state['qha_opt_success']
        qha_registered = scan_state['qha_registered']

        for struct_dir in struct_dirs:
            # 检查每个体积点的优化是否完成
            for vol in self.default_volumes:
                if vol == 1.0:
                    continue

                volume_dir = struct_dir / f'volume_{vol}'
                volume_key = (struct_dir.name, volume_dir.name)
                if volume_key in qha_registered and not volume_dir.exists():
                    self._workflow_clear(struct_dir.name, volume_name=volume_dir.name, stages=['qha_generated'])
                    qha_registered.discard(volume_key)

                opt_dir = volume_dir / 'opt'

                # 如果优化目录不存在，跳过
                if not opt_dir.exists():
                    continue

                # 检查优化是否完成
                if volume_key not in qha_opt_success:
                    continue

                if volume_key in qha_registered:
                    continue

                # 检查声子任务是否已生成
                task_dirs = self._register_submit_tasks(volume_dir, 'qha')
                if task_dirs:
                    qha_registered.add(volume_key)
                    self._workflow_set(struct_dir.name, 'qha_generated', volume_name=volume_dir.name)
                    continue

                # 优化完成但声子任务未生成，现在生成
                logger.info(f"  生成体积 {vol} 的声子任务: {struct_dir.name}")

                # 读取优化后的结构
                contcar = opt_dir / 'CONTCAR'
                if not contcar.exists():
                    logger.warning(f"    找不到CONTCAR: {contcar}")
                    continue

                atoms_optimized = read(str(contcar))

                supercell = self.phonon_config.get('supercell', None)
                max_atoms = self.phonon_config.get('max_atoms', None)
                min_atoms = self.phonon_config.get('min_atoms', 100)
                min_length = self.phonon_config.get('min_length', 10.0)
                distance = self.phonon_config.get('displacement_distance', 0.01)

                n_tasks = generate_phonon_displacements(
                    atoms=atoms_optimized,
                    volume_dir=str(volume_dir),
                    supercell=supercell,
                    max_atoms=max_atoms,
                    min_atoms=min_atoms,
                    min_length=min_length,
                    distance=distance
                )
                logger.info(f"    生成 {n_tasks} 个位移任务")

                # 添加qha任务到队列，包含 task.perfect
                self._register_submit_tasks(volume_dir, 'qha')
                qha_registered.add(volume_key)
                self._workflow_set(struct_dir.name, 'qha_generated', volume_name=volume_dir.name)

                # VASP输入文件在提交时实时生成

    def check_qha_opt_completed(self, struct_dir: Path) -> bool:
        """检查所有QHA体积点的优化是否完成"""
        for vol in self.default_volumes:
            if vol == 1.0:
                continue

            volume_dir = struct_dir / f'volume_{vol}'
            opt_dir = volume_dir / 'opt'

            if not opt_dir.exists():
                return False

            task = self._get_db_task(opt_dir)
            if not task or task['status'] != 'success':
                return False

        return True

    def run_postprocess(self):
        """执行后处理"""
        logger.info("=== 执行后处理 ===")

        for struct_dir in self.get_all_structures():
            # 声子后处理
            if not self._workflow_has(struct_dir.name, 'phonon_postprocessed', volume_name='volume_1.0'):
                if self.check_phonon_completed(struct_dir):
                    volume_dir = struct_dir / 'volume_1.0'
                    analyze_file = volume_dir / 'analyze' / 'phonopy_params.yaml'
                    failure_key = self._postprocess_failure_key(
                        struct_dir, 'phonon', volume_name=volume_dir.name
                    )
                    if not analyze_file.exists() and failure_key not in self._postprocess_failures:
                        logger.info(f"[POSTPROCESS] {struct_dir.name} 开始声子后处理...")
                        try:
                            self._postprocess_phonon(struct_dir)
                            self._postprocess_failures.discard(failure_key)
                            self._clear_postprocess_error(
                                struct_dir, 'phonon', volume_name=volume_dir.name
                            )
                        except Exception as exc:
                            self._postprocess_failures.add(failure_key)
                            self._record_postprocess_error(
                                struct_dir, 'phonon', exc, volume_name=volume_dir.name
                            )
                            logger.exception(f"  声子后处理失败: {struct_dir.name}")

            # QHA后处理
            if not self._workflow_has(struct_dir.name, 'qha_postprocessed'):
                if self._check_all_qha_phonons_completed(struct_dir):
                    volume_postprocess_failed = False
                    for vol in self.default_volumes:
                        volume_dir = struct_dir / f'volume_{vol}'
                        analyze_file = volume_dir / 'analyze' / 'phonopy_params.yaml'
                        failure_key = self._postprocess_failure_key(
                            struct_dir, 'phonon', volume_name=volume_dir.name
                        )
                        if volume_dir.exists() and not analyze_file.exists() and failure_key not in self._postprocess_failures:
                            try:
                                self._postprocess_phonon_volume(struct_dir, volume_dir)
                                self._postprocess_failures.discard(failure_key)
                                self._clear_postprocess_error(
                                    struct_dir, 'phonon', volume_name=volume_dir.name
                                )
                            except Exception as exc:
                                volume_postprocess_failed = True
                                self._postprocess_failures.add(failure_key)
                                self._record_postprocess_error(
                                    struct_dir, 'phonon', exc, volume_name=volume_dir.name
                                )
                                logger.exception(
                                    f"  QHA体积点声子后处理失败: {struct_dir.name}/{volume_dir.name}"
                                )

                    if volume_postprocess_failed:
                        continue

                    static_ready, missing_volumes = self._qha_static_energies_ready(struct_dir)
                    if not static_ready:
                        missing_summary = ', '.join(missing_volumes[:5])
                        if len(missing_volumes) > 5:
                            missing_summary += ', ...'
                        logger.info(
                            f"  跳过QHA后处理: {struct_dir.name} 静态能未齐 "
                            f"({len(missing_volumes)}/{len(self.default_volumes)} 缺失: {missing_summary})"
                        )
                        continue

                    qha_failure_key = self._postprocess_failure_key(struct_dir, 'qha')
                    if qha_failure_key in self._postprocess_failures:
                        continue

                    try:
                        self._postprocess_qha(struct_dir)
                        self._postprocess_failures.discard(qha_failure_key)
                        self._clear_postprocess_error(struct_dir, 'qha')
                    except Exception as exc:
                        self._postprocess_failures.add(qha_failure_key)
                        self._record_postprocess_error(struct_dir, 'qha', exc)
                        logger.exception(f"  QHA后处理失败: {struct_dir.name}")

    def _check_all_qha_phonons_completed(self, struct_dir: Path) -> bool:
        """检查所有QHA体积点的声子计算是否完成"""
        for vol in self.default_volumes:
            volume_dir = struct_dir / f'volume_{vol}'
            if not volume_dir.exists():
                return False

            task_type = 'phonon' if vol == 1.0 else 'qha'
            if not self._check_registered_submit_tasks_completed(volume_dir, task_type):
                return False

        return True

    def _postprocess_phonon(self, struct_dir: Path):
        """声子后处理"""
        volume_dir = struct_dir / 'volume_1.0'
        self._postprocess_phonon_volume(struct_dir, volume_dir)
        # 创建phonon完成标记
        (struct_dir / '.phonon_done').touch()
        self._postprocess_marker(struct_dir, 'phonon__volume_1.0').touch()
        self._workflow_set(struct_dir.name, 'phonon_postprocessed', volume_name=volume_dir.name)
        logger.info(f"  声子后处理完成: {struct_dir.name}")

    def _postprocess_phonon_volume(self, struct_dir: Path, volume_dir: Path):
        """对指定体积目录进行声子后处理"""
        logger.info(f"  声子后处理: {volume_dir}")

        t_min = self.phonon_config.get('t_min', 0)
        t_max = self.phonon_config.get('t_max', 2000)
        t_step = self.phonon_config.get('t_step', 10)

        use_vasprun = (self.worker_mode == 'vasp')

        has_imaginary = postprocess_phonon(
            volume_dir=str(volume_dir),
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            use_vasprun=use_vasprun
        )
        self.imaginary_db.set_cached_result(volume_dir, has_imaginary)

        if has_imaginary:
            logger.warning(f"    存在虚频")
            # 创建虚频标记文件（在结构文件夹下）
            (volume_dir.parent / '.has_imag').touch()
            self._workflow_set(struct_dir.name, 'imaginary_checked', state='has_imaginary', volume_name=volume_dir.name)
        elif volume_dir.name == 'volume_1.0':
            imaginary_flag = volume_dir.parent / '.imaginary_frequency'
            if imaginary_flag.exists():
                imaginary_flag.unlink()
            self._workflow_set(struct_dir.name, 'imaginary_checked', state='clear', volume_name=volume_dir.name)
        else:
            self._workflow_set(struct_dir.name, 'imaginary_checked', state='clear', volume_name=volume_dir.name)

    def _postprocess_qha(self, struct_dir: Path):
        """QHA后处理"""
        logger.info(f"  QHA后处理: {struct_dir.name}")

        pressure = self.qha_config.get('pressure', 0)
        t_min = self.qha_config.get('t_min', 0)
        t_max = self.qha_config.get('t_max', 1000)
        t_step = self.qha_config.get('t_step', 10)

        use_vasprun = (self.worker_mode == 'vasp')

        postprocess_qha(
            struct_dir=str(struct_dir),
            volumes=self.default_volumes,
            pressure=pressure,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            use_vasprun=use_vasprun
        )
        self._postprocess_marker(struct_dir, 'qha').touch()
        self._workflow_set(struct_dir.name, 'qha_postprocessed')

    def _qha_static_energies_ready(self, struct_dir: Path) -> Tuple[bool, List[str]]:
        """检查 QHA 所需静态能是否齐全。"""
        use_vasprun = (self.worker_mode == 'vasp')
        missing_volumes = get_missing_qha_static_energy_volumes(
            str(struct_dir),
            self.default_volumes,
            use_vasprun=use_vasprun,
        )
        return len(missing_volumes) == 0, missing_volumes

    # ========== BTE 工作流 ==========
    # 流程: opt(0GPa) → P_XXGPa/opt(带PSTRESS) → fc2 → 虚频检查 → fc3 → task.BTE → BTE后处理
    # 每个压强点独立: fc2完成后先检查虚频，有虚频则跳过fc3

    def _get_bte_pressures(self):
        """获取 BTE 压强列表 (GPa)"""
        return self.bte_config.get('pressures', [0])

    def generate_bte_pressure_opt_tasks(self):
        """0GPa opt 完成后，为每个压强点生成带 PSTRESS 的 opt 任务"""
        if not self.enable_bte:
            return

        logger.info("=== 生成BTE压强优化任务 ===")

        for struct_dir in self.get_all_structures():
            if not self.check_opt_completed(struct_dir):
                continue

            atoms = self._get_optimized_structure(struct_dir)
            if atoms is None:
                continue

            for pressure in self._get_bte_pressures():
                p_dir = struct_dir / f'P_{pressure:02d}GPa'
                opt_dir = p_dir / 'opt'
                marker = self._generation_marker(struct_dir, f'bte_opt__{p_dir.name}')
                if marker.exists():
                    if opt_dir.exists() or self._get_db_task(opt_dir):
                        continue
                    marker.unlink()

                if opt_dir.exists():
                    # 确保注册到数据库
                    task_path = str(opt_dir.relative_to(self.work_dir))
                    self.db.add_task(task_path, 'bte_opt')
                    marker.touch()
                    continue

                logger.info(f"  创建 {struct_dir.name} P={pressure}GPa opt 任务")
                opt_dir.mkdir(parents=True, exist_ok=True)

                # 复制 0GPa 优化后的结构作为起点
                write(str(opt_dir / 'POSCAR'), atoms, format='vasp', vasp5=True)

                task_path = str(opt_dir.relative_to(self.work_dir))
                self.db.add_task(task_path, 'bte_opt')
                marker.touch()

    def generate_bte_tasks(self):
        """压强 opt 完成后，生成 BTE fc2 位移任务"""
        if not self.enable_bte:
            return

        logger.info("=== 生成BTE fc2任务 ===")

        for struct_dir in self.get_all_structures():
            for pressure in self._get_bte_pressures():
                p_dir = struct_dir / f'P_{pressure:02d}GPa'
                opt_dir = p_dir / 'opt'
                bte_dir = p_dir / 'bte'
                fc2_dir = bte_dir / 'fc2'
                fc2_marker = self._generation_marker(struct_dir, f'bte_fc2__{p_dir.name}')
                if fc2_marker.exists():
                    if self._get_tasks_under(fc2_dir, task_type='bte_fc2'):
                        continue
                    fc2_marker.unlink()

                # 压强 opt 必须完成
                opt_task = self._get_db_task(opt_dir)
                if not opt_dir.exists() or not opt_task or opt_task['status'] != 'success':
                    continue

                cache_key = f"{struct_dir.name}_P{pressure}"
                if cache_key in self._bte_generated:
                    continue

                if not bte_dir.exists():
                    self._prepare_bte_displacements_at_pressure(struct_dir, p_dir)

                if bte_dir.exists():
                    if fc2_dir.exists():
                        self._register_submit_tasks(fc2_dir, 'bte_fc2')

                    fc2_marker.touch()
                    self._bte_generated.add(cache_key)

    def generate_bte_fc3_tasks(self):
        """fc2 完成后检查虚频，无虚频才注册 fc3 任务"""
        if not self.enable_bte:
            return

        logger.info("=== 检查虚频并生成BTE fc3任务 ===")

        for struct_dir in self.get_all_structures():
            for pressure in self._get_bte_pressures():
                p_dir = struct_dir / f'P_{pressure:02d}GPa'
                bte_dir = p_dir / 'bte'
                fc3_dir = bte_dir / 'fc3'
                fc3_marker = self._generation_marker(struct_dir, f'bte_fc3__{p_dir.name}')
                if fc3_marker.exists():
                    if self._get_tasks_under(fc3_dir, task_type='bte_fc3') or (bte_dir / '.has_imaginary').exists():
                        continue
                    fc3_marker.unlink()

                if not bte_dir.exists():
                    continue

                if (bte_dir / '.has_imaginary').exists():
                    continue

                if (bte_dir / '.fc2_checked').exists():
                    continue

                if not self._check_fc_completed(bte_dir / 'fc2'):
                    continue

                logger.info(f"  [BTE] {struct_dir.name} P={pressure}GPa: fc2完成，检查虚频...")

                has_imaginary = self._check_bte_imaginary_at(bte_dir)

                if has_imaginary:
                    logger.warning(f"  [BTE] {struct_dir.name} P={pressure}GPa: 存在虚频，跳过fc3")
                    (bte_dir / '.has_imaginary').touch()
                    fc3_marker.touch()
                    continue

                logger.info(f"  [BTE] {struct_dir.name} P={pressure}GPa: 无虚频，注册fc3任务")
                (bte_dir / '.fc2_checked').touch()

                if fc3_dir.exists():
                    task_dirs = self._register_submit_tasks(fc3_dir, 'bte_fc3')
                    logger.info(f"    注册 {len(task_dirs)} 个fc3任务")
                    fc3_marker.touch()

    def _prepare_bte_displacements_at_pressure(self, struct_dir: Path, p_dir: Path):
        """为指定压强点准备 BTE 位移"""
        struct_name = struct_dir.name
        pressure = p_dir.name  # e.g. P_10GPa

        opt_dir = p_dir / 'opt'
        contcar = opt_dir / 'CONTCAR'
        if contcar.exists() and contcar.stat().st_size > 0:
            atoms = read(str(contcar))
        else:
            atoms = read(str(opt_dir / 'POSCAR'))

        bte_dir = p_dir / 'bte'
        logger.info(f"  生成BTE位移: {struct_name}/{pressure}")

        bte_cfg = self.bte_config
        info = generate_bte_displacements(
            atoms=atoms,
            bte_dir=str(bte_dir),
            supercell=bte_cfg.get('supercell', None),
            max_atoms=bte_cfg.get('max_atoms', None),
            min_atoms=bte_cfg.get('min_atoms', 100),
            min_length=bte_cfg.get('min_length', 10.0),
            distance=bte_cfg.get('displacement_distance', 0.03),
            symprec=bte_cfg.get('symprec', 1e-3),
        )
        logger.info(f"    超胞: {info['supercell']}, {info['n_atoms_sc']} atoms")
        logger.info(f"    fc2: {info['n_fc2']}, fc3: {info['n_fc3']} (待虚频检查)")

    def _check_bte_imaginary_at(self, bte_dir: Path) -> bool:
        """检查指定 bte_dir 的虚频"""
        import json
        from phonopy import Phonopy
        from phono3py import load as phono3py_load
        import numpy as np

        analyze_dir = bte_dir / 'analyze'
        ph3 = phono3py_load(str(analyze_dir / 'phono3py_disp.yaml'), log_level=0)

        use_vasprun = (self.worker_mode == 'vasp')
        from .phonon_utils import collect_bte_forces
        fc2_forces = collect_bte_forces(str(bte_dir), 'fc2', use_vasprun=use_vasprun)
        ph3.phonon_forces = fc2_forces
        ph3.produce_fc2(symmetrize_fc2=True)

        phonon = Phonopy(
            ph3.unitcell,
            supercell_matrix=ph3.phonon_supercell_matrix,
            primitive_matrix=ph3.primitive_matrix,
            is_symmetry=True,
            symprec=ph3.symmetry.tolerance,
        )
        phonon.force_constants = ph3.fc2

        phonon.run_mesh([5, 5, 5])
        freqs_all = phonon.mesh.frequencies.flatten()
        min_freq = float(freqs_all.min())
        has_imaginary = bool(np.any(freqs_all < -0.1))

        result = {
            'has_imaginary': has_imaginary,
            'min_freq_THz': min_freq,
            'n_imaginary': int(np.sum(freqs_all < -0.1)),
        }
        with open(analyze_dir / 'imaginary_check.json', 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"    min_freq={min_freq:.4f} THz, imaginary={has_imaginary}")
        return has_imaginary

    def _check_fc_completed(self, fc_dir: Path) -> bool:
        """检查 fc2 或 fc3 目录下所有任务是否完成"""
        if not fc_dir.exists():
            return False
        task_type = 'bte_fc2' if fc_dir.name == 'fc2' else 'bte_fc3'
        return self._check_registered_submit_tasks_completed(fc_dir, task_type)

    def generate_bte_postprocess_tasks(self):
        """BTE 后处理任务（每个压强点独立）"""
        if not self.enable_bte:
            return

        logger.info("=== 生成BTE后处理任务 ===")

        for struct_dir in self.get_all_structures():
            for pressure in self._get_bte_pressures():
                p_dir = struct_dir / f'P_{pressure:02d}GPa'
                bte_dir = p_dir / 'bte'
                analyze_dir = bte_dir / 'analyze'
                task_dir = analyze_dir / 'task.BTE'
                post_marker = self._generation_marker(struct_dir, f'bte_post__{p_dir.name}')
                if post_marker.exists():
                    if task_dir.exists() or self._get_db_task(task_dir):
                        self._bte_postprocess_generated.add(cache_key)
                        continue
                    post_marker.unlink()

                if not bte_dir.exists():
                    continue

                cache_key = f"{struct_dir.name}_P{pressure}"
                if cache_key in self._bte_postprocess_generated:
                    continue

                # fc2 + fc3 都完成才做后处理
                if not (self._check_fc_completed(bte_dir / 'fc2') and
                        self._check_fc_completed(bte_dir / 'fc3')):
                    continue

                if task_dir.exists():
                    task_path = str(task_dir.relative_to(self.work_dir))
                    self.db.add_task(task_path, 'bte_postprocess')
                    post_marker.touch()
                    self._bte_postprocess_generated.add(cache_key)
                    continue

                logger.info(f"  创建 BTE 后处理任务: {struct_dir.name} P={pressure}GPa")
                analyze_dir.mkdir(parents=True, exist_ok=True)
                task_dir.mkdir(exist_ok=True)

                task_path = str(task_dir.relative_to(self.work_dir))
                self.db.add_task(task_path, 'bte_postprocess')
                post_marker.touch()
                self._bte_postprocess_generated.add(cache_key)

    def run_bte_postprocess(self):
        """兼容旧入口：仅生成 BTE 后处理任务。"""
        self.generate_bte_postprocess_tasks()

    def _print_stats(self, stats: dict):
        """打印统计信息"""
        logger.info("\n--- 当前任务状态 ---")
        for task_type, counts in stats.items():
            pending = counts.get('pending', 0)
            running = counts.get('running', 0)
            success = counts.get('success', 0)
            failed = counts.get('failed', 0)
            logger.info(f"{task_type}: pending={pending}, running={running}, "
                  f"success={success}, failed={failed}")
        logger.info("-------------------\n")

    def prepare_tasks_once(self):
        """按当前配置执行一次任务目录准备，不提交任务。"""
        if self.plain_submit:
            self.sync_plain_submit_tasks()
            return self.collect_statistics()

        struct_dirs = self.get_all_structures()
        self.generate_opt_tasks(struct_dirs)

        if self.enable_qha:
            qha_scan_state = self._build_qha_scan_state()
            self.generate_phonon_tasks(struct_dirs, qha_scan_state)
            self.generate_qha_tasks(struct_dirs, qha_scan_state)
            self.generate_qha_phonon_tasks(struct_dirs, qha_scan_state)

        if self.enable_bte:
            self.generate_bte_pressure_opt_tasks()
            self.generate_bte_tasks()
            self.generate_bte_fc3_tasks()
            self.generate_bte_postprocess_tasks()

        return self.collect_statistics()

    # ========== 主循环 ==========

    def run(self):
        """主循环 - Sbatch模式"""
        logger.info(f"Manager started (Sbatch mode), PID: {os.getpid()}")
        logger.info(f"Structures directory: {self.structures_dir}")
        logger.info(f"Scan interval: {self.scan_interval}s")
        logger.info(f"Workflows: qha={self.enable_qha}, bte={self.enable_bte}")
        logger.info(f"Plain submit: {self.plain_submit}")

        # 启动时仅恢复数据库中 running 的任务，不做全量状态回写
        self.reconcile_tracked_running_tasks()

        # 备份计时器
        last_backup_time = time.time()
        backup_interval = 600  # 10分钟

        while True:
            # 1. 先同步running任务状态
            self.sync_running_tasks()

            # 2. 同步任务执行时间
            self.sync_task_times()

            # 3. 任务准备
            self.prepare_tasks_once()

            # 4. 扫描并提交pending任务
            self.submit_pending_tasks()

            # 5. 执行后处理
            if not self.plain_submit:
                if self.enable_qha:
                    self.run_postprocess()

            # 6. 打印统计信息
            stats = self.collect_statistics()
            self._print_stats(stats)

            # 7. 定期备份数据库
            if time.time() - last_backup_time > backup_interval:
                if self.db.backup():
                    logger.info("数据库已备份")
                last_backup_time = time.time()

            # 等待下一轮扫描
            time.sleep(self.scan_interval)


def main():
    """主函数"""
    config = load_config()
    manager = Manager(config)
    manager.run()


if __name__ == '__main__':
    main()
