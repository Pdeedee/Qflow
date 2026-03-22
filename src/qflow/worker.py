"""Worker脚本 - 支持MatterSim和VASP两种计算模式"""

import os
import sys
import time
import traceback
import subprocess
import numpy as np
from pathlib import Path

from ase.io import read, write

from .utils import load_config, set_task_status, record_failed_task
from .queue_manager import QueueManager


class Worker:
    """Worker执行器 - 支持MatterSim和VASP"""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.config = config
        self.queue = QueueManager(config)

        # 计算模式: mattersim 或 vasp
        self.mode = config.get('worker', {}).get('mode', 'mattersim')
        self.vasp_cmd = config.get('worker', {}).get('vasp_cmd', 'mpirun vasp_std')

        if self.mode == 'mattersim':
            from mattersim.forcefield import MatterSimCalculator
            self.calculator = MatterSimCalculator()
            print("MatterSim calculator initialized")
        else:
            self.calculator = None
            print(f"VASP mode, command: {self.vasp_cmd}")

    def run_task(self, task: dict) -> bool:
        """
        执行单个任务
        返回: True成功，False失败
        """
        task_path = task['path']
        task_type = task['task_type']
        task_dir = Path(task_path)

        print(f"[START] Processing: {task_path}", flush=True)
        print(f"[INFO] Full path: {task_dir.absolute()}", flush=True)

        if not task_dir.exists():
            error_msg = f"Task directory does not exist: {task_path}"
            self._handle_failure(task_path, error_msg)
            return False

        # 设置running状态
        print(f"[STATUS] Setting .running flag: {task_path}", flush=True)
        set_task_status(task_path, 'running', self.config)

        try:
            print(f"[EXEC] Executing {self.mode.upper()} in: {task_path}", flush=True)
            if self.mode == 'mattersim':
                success = self._run_mattersim_task(task_dir, task_type)
            else:
                success = self._run_vasp_task(task_dir, task_type)

            if success:
                print(f"[STATUS] Setting .success flag: {task_path}", flush=True)
                set_task_status(task_path, 'success', self.config)
                print(f"[QUEUE] Moving to done queue: {task_path}", flush=True)
                self.queue.update_task_status(task_path, 'success')
                print(f"[SUCCESS] Completed: {task_path}", flush=True)
                return True
            else:
                return False

        except Exception as e:
            error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
            self._handle_failure(task_path, error_msg)
            return False

    # ========== MatterSim 模式 ==========

    def _run_mattersim_task(self, task_dir: Path, task_type: str) -> bool:
        """使用MatterSim执行任务"""
        if task_type == 'opt':
            return self._run_mattersim_opt(task_dir)
        elif task_type == 'phonon':
            return self._run_mattersim_phonon(task_dir)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _run_mattersim_opt(self, task_dir: Path) -> bool:
        """MatterSim结构优化"""
        from ase.optimize import BFGS

        poscar = task_dir / 'POSCAR'
        if not poscar.exists():
            raise FileNotFoundError(f"POSCAR not found in {task_dir}")

        atoms = read(str(poscar))
        atoms.calc = self.calculator

        fmax = self.config.get('opt', {}).get('fmax', 0.01)
        log_file = task_dir / 'opt.log'

        with open(log_file, 'w') as f:
            opt = BFGS(atoms, logfile=f)
            opt.run(fmax=fmax, steps=500)

        write(str(task_dir / 'CONTCAR'), atoms, format='vasp', vasp5=True)

        energy = atoms.get_potential_energy()
        with open(task_dir / 'energy.txt', 'w') as f:
            f.write(f"{energy:.8f}\n")

        return True

    def _run_mattersim_phonon(self, task_dir: Path) -> bool:
        """MatterSim力计算"""
        poscar = task_dir / 'POSCAR'
        if not poscar.exists():
            raise FileNotFoundError(f"POSCAR not found in {task_dir}")

        atoms = read(str(poscar))
        atoms.calc = self.calculator

        forces = atoms.get_forces()
        np.savetxt(str(task_dir / 'forces.txt'), forces, fmt='%.8f')

        energy = atoms.get_potential_energy()
        with open(task_dir / 'energy.txt', 'w') as f:
            f.write(f"{energy:.8f}\n")

        return True

    # ========== VASP 模式 ==========

    def _run_vasp_task(self, task_dir: Path, task_type: str) -> bool:
        """使用VASP执行任务"""
        # VASP模式：直接运行vasp命令
        # 显式传递环境变量以确保 PATH 和 LD_LIBRARY_PATH 正确
        import os
        env = os.environ.copy()

        result = subprocess.run(
            self.vasp_cmd,
            shell=True,
            cwd=task_dir,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            error_msg = f"VASP failed with exit code {result.returncode}\n"
            error_msg += f"STDERR: {result.stderr}"
            raise RuntimeError(error_msg)

        return True

    # ========== 通用方法 ==========

    def _handle_failure(self, task_path: str, error_msg: str):
        """处理任务失败"""
        print(f"[STATUS] Setting .failed flag: {task_path}")
        set_task_status(task_path, 'failed', self.config, error_msg)
        print(f"[QUEUE] Moving to failed queue: {task_path}")
        self.queue.update_task_status(task_path, 'failed')
        record_failed_task(task_path, self.config)
        print(f"[FAILED] {task_path}")
        print(f"  Error: {error_msg[:200]}...")

    def run(self, max_idle_time: int = 60):
        """
        主循环：不断获取任务并执行

        Args:
            max_idle_time: 最大空闲时间(秒)，超过则退出
        """
        import random

        print(f"Worker started, PID: {os.getpid()}", flush=True)
        print(f"Mode: {self.mode}", flush=True)
        print(f"Working directory: {os.getcwd()}", flush=True)

        # 随机休眠5-20秒，避免多个worker同时启动时抢同一个任务
        sleep_time = random.uniform(5, 20)
        print(f"Initial sleep for {sleep_time:.1f}s to avoid race condition...", flush=True)
        time.sleep(sleep_time)
        print(f"Ready to process tasks", flush=True)

        idle_time = 0
        check_interval = 5

        while True:
            task = self.queue.get_pending_task()

            if task is None:
                idle_time += check_interval
                if idle_time >= max_idle_time:
                    print(f"No tasks for {max_idle_time}s, exiting...", flush=True)
                    break
                print(f"No pending tasks, waiting... ({idle_time}s)", flush=True)
                time.sleep(check_interval)
                continue

            idle_time = 0
            print(f"\n{'='*60}", flush=True)
            print(f"[TASK] Claimed from queue: {task['path']}", flush=True)
            print(f"[TASK] Type: {task['task_type']}, Priority: {task.get('priority', 0)}", flush=True)
            print(f"[TASK] Working on directory: {task['path']}", flush=True)
            print(f"{'='*60}", flush=True)
            self.run_task(task)
            print(f"{'='*60}\n", flush=True)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='QFlow Worker')
    parser.add_argument('--max-idle', type=int, default=60,
                        help='Max idle time in seconds before exit (default: 60)')
    parser.add_argument('--mode', type=str, choices=['mattersim', 'vasp'],
                        help='Override calculation mode from config')
    args = parser.parse_args()

    config = load_config()

    # 命令行参数覆盖配置
    if args.mode:
        config.setdefault('worker', {})['mode'] = args.mode

    worker = Worker(config)
    worker.run(max_idle_time=args.max_idle)


if __name__ == '__main__':
    main()
