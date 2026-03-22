"""Manager管理进程 - Sbatch模式，直接提交任务到SLURM"""

import os
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Set, Optional, Dict
from datetime import datetime

from .logger import logger

logger.info("正在加载 ASE...")
from ase.io import read, write

logger.info("正在加载 qflow 模块...")
from .utils import load_config, get_structure_name
from .template import generate_task_script
from .task_db import TaskDB
from .phonon_utils import (
    generate_phonon_displacements,
    postprocess_phonon,
    check_imaginary_frequency,
    postprocess_qha
)

logger.info("正在加载 pymatgen...")
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# VASP输入文件生成
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MatPESStaticSet
from pymatgen.core import Structure as PMGStructure

logger.info("所有模块加载完成")


class Manager:
    """任务管理器 - Sbatch模式"""

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

        # Worker配置
        self.worker_mode = config.get('worker', {}).get('mode', 'mattersim')

        # 结构优化配置
        opt_config = config.get('opt', {})
        self.refine_structure = opt_config.get('refine_structure', False)

        # INCAR设置在每次提交任务时实时从config.yaml读取（见_generate_vasp_inputs）

        # POTCAR设置
        potcar_config = config.get('potcar', {})
        self.potcar_functional = potcar_config.get('functional', 'PBE_54')

        # 默认体积列表 (用于QHA)
        self.default_volumes = self.qha_config.get('volumes', [0.98, 0.99, 1.0, 1.01, 1.02])

        # SQLite任务数据库
        logger.info("正在初始化任务数据库...")
        self.db = TaskDB(config)

        # 标记已处理的结构，避免重复
        self._phonon_generated: Set[str] = set()
        self._qha_generated: Set[str] = set()
        self._phonon_postprocessed: Set[str] = set()
        self._qha_postprocessed: Set[str] = set()

        logger.info("Manager 初始化完成")

    # ========== 任务状态管理 ==========

    def get_task_status(self, task_path: Path) -> str:
        """检查任务状态（基于文件系统标记）

        Returns: 'success', 'failed', 'running', 'pending', 'not_ready'
        """
        if (task_path / '.success').exists():
            return 'success'
        if (task_path / '.failed').exists():
            return 'failed'
        if (task_path / '.running').exists():
            return 'running'
        # 检查是否有POSCAR（表示任务已准备好）
        if (task_path / 'POSCAR').exists():
            return 'pending'
        return 'not_ready'

    def record_task_time(self, task_path: Path, start_time: str, end_time: str, duration: float, status: str):
        """记录任务执行时间

        Args:
            task_path: 任务路径（相对于work_dir）
            start_time: 开始时间 ISO格式
            end_time: 结束时间 ISO格式
            duration: 持续时间（秒）
            status: 任务状态（success/failed）
        """
        try:
            # 生成记录文件名（使用任务路径的hash）
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

        except Exception as e:
            logger.warning(f"记录任务时间失败 {task_path}: {e}")

    def sync_task_times(self):
        """同步任务执行时间（从SLURM日志文件中提取）"""
        try:
            for struct_dir in self.get_all_structures():
                # 检查opt任务
                opt_dir = struct_dir / 'opt'
                if opt_dir.exists():
                    self._extract_task_time(opt_dir)

                # 检查phonon/qha任务
                for volume_dir in struct_dir.glob('volume_*'):
                    for task_dir in volume_dir.glob('task.*'):
                        if task_dir.is_dir():
                            self._extract_task_time(task_dir)

        except Exception as e:
            logger.error(f"同步任务时间失败: {e}")

    def _extract_task_time(self, task_dir: Path):
        """从任务目录的.task_time文件提取执行时间并更新到队列"""
        status = self.get_task_status(task_dir)
        if status not in ['success', 'failed']:
            return

        # 获取任务相对路径
        try:
            task_rel = str(task_dir.relative_to(self.work_dir))
        except:
            return

        # 检查是否已记录过（避免重复记录）
        task_hash = abs(hash(task_rel)) % 1000000
        record_file = self.task_times_dir / f"{task_hash:06d}.json"
        if record_file.exists():
            return  # 已记录过

        # 优先读取 .task_time 文件
        task_time_file = task_dir / '.task_time'
        if task_time_file.exists():
            try:
                content = task_time_file.read_text()
                # 解析YAML格式
                time_data = {}
                for line in content.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        time_data[key.strip()] = value.strip()

                start_time = time_data.get('start_time', '')
                end_time = time_data.get('end_time', '')
                duration = float(time_data.get('duration_seconds', 0))
                task_status = time_data.get('status', status)

                if start_time and end_time:
                    # 记录到文件
                    self.record_task_time(task_dir, start_time, end_time, duration, task_status)
                    # 同时更新队列数据库
                    self.db.update_task_time(task_rel, start_time, end_time, duration, task_status)
                    return

            except Exception as e:
                logger.debug(f"解析.task_time失败 {task_dir}: {e}")

        # 回退：从SLURM日志文件中解析时间
        slurm_logs = list(task_dir.glob('slurm_*.log'))
        if not slurm_logs:
            return

        log_file = max(slurm_logs, key=lambda p: p.stat().st_mtime)

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            start_time = None
            end_time = None

            for line in lines:
                if 'Date:' in line and start_time is None:
                    try:
                        date_str = line.split('Date:')[-1].strip()
                        start_time = datetime.strptime(date_str, '%a %b %d %I:%M:%S %p %Z %Y').isoformat()
                    except:
                        pass

            end_time = datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()

            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
                duration = (end_dt - start_dt).total_seconds()

                # 记录到文件
                self.record_task_time(task_dir, start_time, end_time, duration, status)
                # 同时更新队列数据库
                self.db.update_task_time(task_rel, start_time, end_time, duration, status)

        except Exception as e:
            logger.debug(f"提取任务时间失败 {task_dir}: {e}")

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

                    # 检查文件系统状态
                    if (opt_dir / '.success').exists():
                        self.db.update_status(task_path, 'success')
                        synced_counts['updated_success'] += 1
                    elif (opt_dir / '.failed').exists():
                        self.db.update_status(task_path, 'failed')
                        synced_counts['updated_failed'] += 1
                    elif (opt_dir / '.running').exists():
                        self.db.update_status(task_path, 'running')
                        synced_counts['updated_running'] += 1
                    else:
                        # pending状态，添加到队列（如果不存在）
                        if self.db.add_task(task_path, 'opt'):
                            synced_counts['added'] += 1

                # 扫描phonon/qha任务
                for volume_dir in struct_dir.glob('volume_*'):
                    for task_dir in volume_dir.glob('task.*'):
                        if not task_dir.is_dir() or task_dir.name == 'task_perfect':
                            continue
                        if not (task_dir / 'POSCAR').exists():
                            continue

                        task_path = str(task_dir.relative_to(self.work_dir))
                        all_task_paths.add(task_path)

                        # 判断任务类型
                        task_type = 'qha' if volume_dir.name != 'volume_1.0' else 'phonon'

                        # 检查文件系统状态
                        if (task_dir / '.success').exists():
                            self.db.update_status(task_path, 'success')
                            synced_counts['updated_success'] += 1
                        elif (task_dir / '.failed').exists():
                            self.db.update_status(task_path, 'failed')
                            synced_counts['updated_failed'] += 1
                        elif (task_dir / '.running').exists():
                            self.db.update_status(task_path, 'running')
                            synced_counts['updated_running'] += 1
                        else:
                            # pending状态
                            if self.db.add_task(task_path, task_type):
                                synced_counts['added'] += 1

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

    def sync_running_tasks_status(self):
        """快速同步：只检查running任务的文件系统状态"""
        synced = 0
        reset_to_failed = 0
        reset_to_pending = 0
        running_tasks = self.db.get_running_tasks()

        # 获取当前所有活跃的SLURM job ID
        active_jobs = set()
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'root'), '-h', '-o', '%i'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                active_jobs = set(result.stdout.strip().split())
        except Exception:
            pass  # squeue失败时跳过SLURM检查

        for task_data in running_tasks:
            task_path_str = task_data['path']
            task_path = self.work_dir / task_path_str

            if not task_path.exists():
                self.db.update_status(task_path_str, 'pending')
                reset_to_pending += 1
                continue

            # 检查文件系统状态
            if (task_path / '.success').exists():
                self.db.update_status(task_path_str, 'success')
                synced += 1
            elif (task_path / '.failed').exists():
                self.db.update_status(task_path_str, 'failed')
                synced += 1
            elif not (task_path / '.running').exists():
                self.db.update_status(task_path_str, 'pending')
                reset_to_pending += 1
                logger.warning(f"任务 {task_path_str} 无运行标记，重置为pending")
            elif active_jobs:
                # .running存在，检查SLURM job是否还活着
                slurm_job_id = task_data.get('slurm_job_id', '')
                if slurm_job_id and slurm_job_id not in active_jobs:
                    # SLURM job已消失但没有.success/.failed，任务异常终止
                    (task_path / '.running').unlink(missing_ok=True)
                    (task_path / '.failed').touch()
                    self.db.update_status(task_path_str, 'failed')
                    reset_to_failed += 1
                    logger.warning(f"任务 {task_path_str} SLURM job {slurm_job_id} 已消失，标记为failed")

        if synced > 0:
            logger.info(f"同步了 {synced} 个running任务状态")
        if reset_to_failed > 0:
            logger.info(f"标记了 {reset_to_failed} 个异常终止任务为failed")
        if reset_to_pending > 0:
            logger.info(f"重置了 {reset_to_pending} 个异常任务为pending")

    # ========== Sbatch任务提交 ==========

    def _generate_task_name(self, task_dir: Path) -> str:
        """生成任务名称用于sbatch job-name

        例如: opt_mp-1234, phonon_mp-1234_vol1.0_task001
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

    def submit_sbatch_task(self, task_dir: Path) -> Optional[str]:
        """为任务生成sbatch脚本并提交

        Returns: job_id或None（失败时）
        """
        # 生成任务名称
        task_name = self._generate_task_name(task_dir)

        # 生成sbatch脚本
        script_content = generate_task_script(self.config, task_name)

        # 写入脚本文件到任务目录
        script_file = task_dir / 'run.sbatch'
        script_file.write_text(script_content)
        script_file.chmod(0o755)

        # 创建.running标记
        (task_dir / '.running').touch()

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
            (task_dir / '.running').unlink()  # 删除.running标记
            return None

    def _save_job_mapping(self, job_id: str, task_path: Path):
        """保存job_id到任务路径的映射"""
        jobs = {}
        if self.jobs_file.exists():
            try:
                jobs = json.loads(self.jobs_file.read_text())
            except:
                jobs = {}

        jobs[job_id] = str(task_path.relative_to(self.work_dir))
        self.jobs_file.write_text(json.dumps(jobs, indent=2))

    def submit_pending_tasks(self):
        """从队列获取pending任务并提交（受max_workers限制）"""
        # 读取最大并发任务数，默认为1
        max_workers = 1  # 默认值
        if self.max_workers_file.exists():
            try:
                max_workers = int(self.max_workers_file.read_text().strip())
            except:
                logger.warning(f"无法读取 {self.max_workers_file}，使用默认值 {max_workers}")

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

            if not task_dir.exists() or not (task_dir / 'POSCAR').exists():
                # 任务目录不存在，标记为failed
                self.db.update_status(task_path_str, 'failed')
                logger.warning(f"  任务目录不存在，标记为failed: {task_path_str}")
                continue

            # 提交前实时生成VASP输入文件（INCAR/POTCAR/KPOINTS）
            task_type = task_data.get('task_type', 'opt')
            vasp_type = task_type if task_type in ('opt', 'phonon', 'qha_opt') else 'phonon'
            if self.worker_mode == 'vasp':
                try:
                    self._generate_vasp_inputs(task_dir, task_type=vasp_type)
                except Exception as e:
                    logger.error(f"  生成VASP输入失败: {task_path_str} - {e}")
                    self.db.update_status(task_path_str, 'failed')
                    continue

            # 提交sbatch任务
            job_id = self.submit_sbatch_task(task_dir)
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

    def generate_opt_tasks(self):
        """生成结构优化任务"""
        logger.info("=== 扫描结构优化任务 ===")

        for struct_dir in self.get_all_structures():
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

    def check_opt_completed(self, struct_dir: Path) -> bool:
        """检查结构优化是否完成"""
        opt_dir = struct_dir / 'opt'
        if not opt_dir.exists():
            return False
        return self.get_task_status(opt_dir) == 'success'

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

    def generate_phonon_tasks(self):
        """生成声子计算任务"""
        logger.info("=== 生成声子计算任务 ===")

        for struct_dir in self.get_all_structures():
            struct_name = struct_dir.name

            if struct_name in self._phonon_generated:
                continue

            if not self.check_opt_completed(struct_dir):
                continue

            volume_dir = struct_dir / 'volume_1.0'

            if not volume_dir.exists():
                self._prepare_phonon_tasks(struct_dir, volume_dir)

            if volume_dir.exists():
                task_dirs = [d for d in volume_dir.iterdir()
                            if d.is_dir() and d.name.startswith('task.') and d.name != 'task_perfect']

                # 确保已存在的任务都注册到数据库
                for task_dir in task_dirs:
                    task_path = str(task_dir.relative_to(self.work_dir))
                    self.db.add_task(task_path, 'phonon')

                if task_dirs:
                    self._phonon_generated.add(struct_name)

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

        try:
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

            # 添加phonon任务到队列
            for i in range(n_tasks):
                task_dir = volume_dir / f'task.{i:06d}'
                task_path = str(task_dir.relative_to(self.work_dir))
                self.db.add_task(task_path, 'phonon')

            # 声子位移任务的VASP输入文件在提交时实时生成

        except Exception as e:
            logger.error(f"  生成声子任务失败 - {e}")

    def _generate_vasp_inputs(self, task_dir: Path, task_type: str = 'opt'):
        """生成VASP输入文件

        每次调用时实时读取 config.yaml 的 incar 设置，确保使用最新配置。

        Args:
            task_dir: 任务目录
            task_type: 任务类型
                - 'opt': 结构优化，使用 MPRelaxSet + ISIF=3
                - 'qha_opt': QHA体积点优化，使用 MPRelaxSet + ISIF=2
                - 'phonon': 声子单点计算，使用 MPStaticSet
        """
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='pymatgen')

        poscar = task_dir / 'POSCAR'
        if not poscar.exists():
            return

        # 实时读取最新的 config.yaml incar 设置
        fresh_config = load_config()
        incar_config = fresh_config.get('incar', {})
        incar_opt = incar_config.get('opt', {})
        incar_phonon = incar_config.get('phonon', {})
        incar_qha_opt = incar_config.get('qha_opt', {})

        # 兼容旧配置格式
        if not incar_opt and not incar_phonon:
            incar_opt = incar_config
            incar_phonon = incar_config

        # 如果没有配置qha_opt，使用opt的配置但修改ISIF=2
        if not incar_qha_opt and incar_opt:
            incar_qha_opt = incar_opt.copy()
            incar_qha_opt['ISIF'] = 2

        structure = PMGStructure.from_file(str(poscar))

        if task_type == 'phonon':
            # 声子计算使用 MPStaticSet（单点计算）
            incar_settings = incar_phonon
            mp_set = MatPESStaticSet(structure, user_incar_settings=incar_settings,
                                    user_potcar_functional=self.potcar_functional)
        elif task_type == 'qha_opt':
            # QHA体积点优化使用 MPRelaxSet + ISIF=2
            incar_settings = incar_qha_opt
            mp_set = MPRelaxSet(structure, user_incar_settings=incar_settings,
                               user_potcar_functional=self.potcar_functional)
        else:
            # 结构优化使用 MPRelaxSet + ISIF=3
            incar_settings = incar_opt
            mp_set = MPRelaxSet(structure, user_incar_settings=incar_settings,
                               user_potcar_functional=self.potcar_functional)

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

        task_dirs = [d for d in volume_dir.iterdir()
                     if d.is_dir() and d.name.startswith('task.') and d.name != 'task_perfect']

        if not task_dirs:
            return False

        for task_dir in task_dirs:
            if self.get_task_status(task_dir) != 'success':
                return False

        return True

    def check_imaginary_frequency_wrapper(self, struct_dir: Path) -> bool:
        """检查是否存在虚频"""
        volume_dir = struct_dir / 'volume_1.0'
        if not volume_dir.exists():
            return False

        try:
            return check_imaginary_frequency(str(volume_dir))
        except Exception as e:
            logger.warning(f"  虚频检查失败 - {e}")
            return False

    def generate_qha_tasks(self):
        """生成QHA任务"""
        logger.info("=== 生成QHA计算任务 ===")

        for struct_dir in self.get_all_structures():
            struct_name = struct_dir.name

            if struct_name in self._qha_generated:
                continue

            volume_1_analyze = struct_dir / 'volume_1.0' / 'analyze' / 'phonopy_params.yaml'
            if not volume_1_analyze.exists():
                continue

            if self.check_imaginary_frequency_wrapper(struct_dir):
                logger.info(f"跳过 {struct_name}：存在虚频")
                imaginary_flag = struct_dir / '.imaginary_frequency'
                imaginary_flag.touch()
                self._qha_generated.add(struct_name)
                continue

            self._prepare_qha_volumes(struct_dir)

            all_volumes_ready = True
            for vol in self.default_volumes:
                volume_dir = struct_dir / f'volume_{vol}'
                if not volume_dir.exists():
                    all_volumes_ready = False
                    break

            if all_volumes_ready:
                self._qha_generated.add(struct_name)

    def _prepare_qha_volumes(self, struct_dir: Path):
        """准备QHA所需的各体积点 - 第一步：创建优化任务"""
        atoms = self._get_optimized_structure(struct_dir)
        if atoms is None:
            return

        for vol in self.default_volumes:
            if vol == 1.0:
                continue

            volume_dir = struct_dir / f'volume_{vol}'
            opt_dir = volume_dir / 'opt'

            # 如果优化目录已存在，跳过
            if opt_dir.exists():
                # 确保优化任务已注册到数据库
                task_path = str(opt_dir.relative_to(self.work_dir))
                self.db.add_task(task_path, 'qha_opt')
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
            logger.info(f"    已创建优化任务: {task_path}")

    def generate_qha_phonon_tasks(self):
        """生成QHA声子任务 - 在优化完成后"""
        logger.info("=== 检查QHA优化并生成声子任务 ===")

        for struct_dir in self.get_all_structures():
            struct_name = struct_dir.name

            # 检查每个体积点的优化是否完成
            for vol in self.default_volumes:
                if vol == 1.0:
                    continue

                volume_dir = struct_dir / f'volume_{vol}'
                opt_dir = volume_dir / 'opt'

                # 如果优化目录不存在，跳过
                if not opt_dir.exists():
                    continue

                # 检查优化是否完成
                if self.get_task_status(opt_dir) != 'success':
                    continue

                # 检查声子任务是否已生成
                task_dirs = list(volume_dir.glob('task.*'))
                if any(d.name != 'task_perfect' for d in task_dirs if d.is_dir()):
                    # 声子任务已存在，确保都注册到数据库
                    for task_dir in task_dirs:
                        if task_dir.is_dir() and task_dir.name != 'task_perfect':
                            task_path = str(task_dir.relative_to(self.work_dir))
                            self.db.add_task(task_path, 'qha')
                    continue

                # 优化完成但声子任务未生成，现在生成
                logger.info(f"  生成体积 {vol} 的声子任务: {struct_name}")

                # 读取优化后的结构
                contcar = opt_dir / 'CONTCAR'
                if not contcar.exists():
                    logger.warning(f"    找不到CONTCAR: {contcar}")
                    continue

                try:
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

                    # 添加qha任务到队列
                    for i in range(n_tasks):
                        task_dir = volume_dir / f'task.{i:06d}'
                        task_path = str(task_dir.relative_to(self.work_dir))
                        self.db.add_task(task_path, 'qha')

                    # VASP输入文件在提交时实时生成

                except Exception as e:
                    logger.error(f"  生成体积 {vol} 声子任务失败 - {e}")

    def check_qha_opt_completed(self, struct_dir: Path) -> bool:
        """检查所有QHA体积点的优化是否完成"""
        for vol in self.default_volumes:
            if vol == 1.0:
                continue

            volume_dir = struct_dir / f'volume_{vol}'
            opt_dir = volume_dir / 'opt'

            if not opt_dir.exists():
                return False

            if self.get_task_status(opt_dir) != 'success':
                return False

        return True

    def run_postprocess(self):
        """执行后处理"""
        logger.info("=== 执行后处理 ===")

        for struct_dir in self.get_all_structures():
            struct_name = struct_dir.name

            # 声子后处理
            if struct_name not in self._phonon_postprocessed:
                if self.check_phonon_completed(struct_dir):
                    volume_dir = struct_dir / 'volume_1.0'
                    analyze_file = volume_dir / 'analyze' / 'phonopy_params.yaml'
                    if not analyze_file.exists():
                        logger.info(f"[POSTPROCESS] {struct_name} 开始声子后处理...")
                        self._postprocess_phonon(struct_dir)
                    self._phonon_postprocessed.add(struct_name)

            # QHA后处理
            if struct_name not in self._qha_postprocessed:
                if self._check_all_qha_phonons_completed(struct_dir):
                    for vol in self.default_volumes:
                        volume_dir = struct_dir / f'volume_{vol}'
                        analyze_file = volume_dir / 'analyze' / 'phonopy_params.yaml'
                        if volume_dir.exists() and not analyze_file.exists():
                            self._postprocess_phonon_volume(struct_dir, volume_dir)

                    self._postprocess_qha(struct_dir)
                    self._qha_postprocessed.add(struct_name)

    def _check_all_qha_phonons_completed(self, struct_dir: Path) -> bool:
        """检查所有QHA体积点的声子计算是否完成"""
        for vol in self.default_volumes:
            volume_dir = struct_dir / f'volume_{vol}'
            if not volume_dir.exists():
                return False

            task_dirs = [d for d in volume_dir.iterdir()
                         if d.is_dir() and d.name.startswith('task.') and d.name != 'task_perfect']

            if not task_dirs:
                return False

            for task_dir in task_dirs:
                if self.get_task_status(task_dir) != 'success':
                    return False

        return True

    def _postprocess_phonon(self, struct_dir: Path):
        """声子后处理"""
        volume_dir = struct_dir / 'volume_1.0'
        self._postprocess_phonon_volume(struct_dir, volume_dir)
        # 创建phonon完成标记
        (struct_dir / '.phonon_done').touch()
        logger.info(f"  声子后处理完成: {struct_dir.name}")

    def _postprocess_phonon_volume(self, struct_dir: Path, volume_dir: Path):
        """对指定体积目录进行声子后处理"""
        logger.info(f"  声子后处理: {volume_dir}")

        try:
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

            if has_imaginary:
                logger.warning(f"    存在虚频")
                # 创建虚频标记文件（在结构文件夹下）
                (volume_dir.parent / '.has_imag').touch()

        except Exception as e:
            logger.error(f"  声子后处理失败 - {e}")

    def _postprocess_qha(self, struct_dir: Path):
        """QHA后处理"""
        logger.info(f"  QHA后处理: {struct_dir.name}")

        try:
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

        except Exception as e:
            logger.error(f"  QHA后处理失败 - {e}")

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

    # ========== 主循环 ==========

    def run(self):
        """主循环 - Sbatch模式"""
        logger.info(f"Manager started (Sbatch mode), PID: {os.getpid()}")
        logger.info(f"Structures directory: {self.structures_dir}")
        logger.info(f"Scan interval: {self.scan_interval}s")
        logger.info(f"Task times directory: {self.task_times_dir}")

        # 启动时执行一次完整同步
        self.sync_queue_from_filesystem()

        # 备份计时器
        last_backup_time = time.time()
        backup_interval = 600  # 10分钟

        while True:
            try:
                # 1. 先同步running任务状态（让刚完成的任务从running变成success/failed）
                self.sync_running_tasks()

                # 2. 同步任务执行时间
                self.sync_task_times()

                # 3. 生成各阶段任务目录
                self.generate_opt_tasks()
                self.generate_phonon_tasks()
                self.generate_qha_tasks()
                self.generate_qha_phonon_tasks()  # 新增：在QHA优化完成后生成声子任务

                # 4. 扫描并提交pending任务（使用最新的running数）
                self.submit_pending_tasks()

                # 5. 执行后处理
                self.run_postprocess()

                # 6. 打印统计信息
                stats = self.collect_statistics()
                self._print_stats(stats)

                # 7. 定期备份数据库（每10分钟）
                if time.time() - last_backup_time > backup_interval:
                    if self.db.backup():
                        logger.info("数据库已备份")
                    last_backup_time = time.time()

            except Exception as e:
                logger.error(f"Error in manager loop: {e}", exc_info=True)

            # 等待下一轮扫描
            time.sleep(self.scan_interval)


def main():
    """主函数"""
    config = load_config()
    manager = Manager(config)
    manager.run()


if __name__ == '__main__':
    main()
