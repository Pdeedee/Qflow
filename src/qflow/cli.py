#!/usr/bin/env python3
"""qflow - 计算任务工作流管理系统CLI"""

import argparse
import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path

from .utils import load_config
from .template import generate_manager_script


def cmd_worker(args):
    """设置最大并发任务数"""
    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()
    max_workers_file = work_dir / 'max_workers.txt'

    try:
        n = int(args.action)
    except ValueError:
        print(f"无效的参数: {args.action}")
        print("用法: qflow worker <N>  # 设置最大并发任务数为 N")
        return

    # 写入最大并发数到文件
    max_workers_file.write_text(str(n))
    print(f"✓ 已设置最大并发任务数: {n}")
    print(f"  配置文件: {max_workers_file}")
    print()
    print("Manager 会自动控制任务提交数量，确保同时运行的任务不超过此限制")
    print()

    # 显示当前运行的任务数
    result = subprocess.run(
        "squeue -u $USER -o '%.18i %.20j' -h",
        shell=True,
        capture_output=True,
        text=True
    )

    current_running = 0
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                job_name = parts[1]
                if job_name.startswith('opt_') or job_name.startswith('ph_') or job_name.startswith('qflow_task'):
                    current_running += 1

    print(f"当前状态:")
    print(f"  运行任务数: {current_running}")
    print(f"  最大并发数: {n}")
    print()

    if current_running < n:
        print(f"✓ Manager 会继续提交任务直到达到 {n} 个")
    elif current_running == n:
        print(f"✓ 已达到限制，Manager 会等待任务完成后再提交新任务")
    else:
        print(f"⚠ 当前运行数超过限制，Manager 会等待任务完成")


def _get_manager_mode():
    """获取 manager 运行模式"""
    config = load_config()
    return config.get('manager', {}).get('mode', 'sbatch'), config


def _is_manager_running(config=None):
    """检查 manager 是否在运行，返回 (running: bool, info: str)"""
    if config is None:
        _, config = _get_manager_mode()
    mode = config.get('manager', {}).get('mode', 'sbatch')
    work_dir = Path(config.get('work_dir', '.')).resolve()

    if mode == 'local':
        pid_file = work_dir / 'manager.pid'
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                return True, f"本地进程 PID: {pid}"
            except (ProcessLookupError, ValueError):
                pid_file.unlink()
        return False, ""
    else:
        result = subprocess.run(
            "squeue -u $USER -n qflow_manager -o '%.18i' -h",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            ids = result.stdout.strip().split()
            if ids:
                return True, f"SLURM 作业: {', '.join(ids)}"
        return False, ""


def _cancel_manager(config=None):
    """取消 manager（自动识别模式）"""
    if config is None:
        _, config = _get_manager_mode()
    mode = config.get('manager', {}).get('mode', 'sbatch')
    work_dir = Path(config.get('work_dir', '.')).resolve()

    if mode == 'local':
        pid_file = work_dir / 'manager.pid'
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                print(f"✓ 已停止 Manager (PID: {pid})")
                pid_file.unlink()
                return True
            except ProcessLookupError:
                print("Manager 进程已不存在")
                pid_file.unlink()
            except ValueError:
                print("PID 文件内容无效")
                pid_file.unlink()
        else:
            print("没有找到运行中的 Manager")
        return False
    else:
        result = subprocess.run(
            "squeue -u $USER -n qflow_manager -o '%.18i' -h",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            manager_ids = result.stdout.strip().split()
            if manager_ids:
                print(f"找到 {len(manager_ids)} 个 Manager 作业")
                for job_id in manager_ids:
                    subprocess.run(f"scancel {job_id}", shell=True)
                    print(f"  已取消 Manager 作业 {job_id}")
                return True
            else:
                print("没有找到运行中的 Manager")
        else:
            print("查询SLURM作业失败")
        return False


def cmd_manager(args):
    """Manager管理命令"""
    if args.action == 'run':
        # 检查是否在 SLURM 作业中
        import os
        if 'SLURM_JOB_ID' in os.environ:
            # 在 SLURM 作业中，直接运行 manager
            from .manager import Manager
            config = load_config()
            manager = Manager(config)
            manager.run()
        else:
            config = load_config()
            work_dir = Path(config.get('work_dir', '.')).resolve()
            manager_mode = config.get('manager', {}).get('mode', 'sbatch')

            # 创建目录
            subs_dir = work_dir / 'subs'
            log_dir = work_dir / 'log'
            subs_dir.mkdir(exist_ok=True)
            log_dir.mkdir(exist_ok=True)

            if manager_mode == 'local':
                # 本地 nohup 模式
                python_path = sys.executable
                log_file = log_dir / 'manager.log'
                pid_file = work_dir / 'manager.pid'

                # 检查是否已有 manager 在运行
                if pid_file.exists():
                    try:
                        old_pid = int(pid_file.read_text().strip())
                        os.kill(old_pid, 0)  # 检查进程是否存在
                        print(f"Manager 已在运行 (PID: {old_pid})")
                        print(f"  日志: {log_file}")
                        print(f"  停止: qflow manager cancel")
                        return
                    except (ProcessLookupError, ValueError):
                        pid_file.unlink()

                # 启动 nohup 进程
                cmd = f"nohup {python_path} -m qflow.manager > {log_file} 2>&1 & echo $!"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=work_dir
                )

                if result.returncode == 0:
                    pid = result.stdout.strip()
                    pid_file.write_text(pid)
                    print(f"✓ Manager 已在本地启动 (nohup)")
                    print(f"  PID: {pid}")
                    print(f"  日志: {log_file}")
                    print(f"  停止: qflow manager cancel")
                else:
                    print(f"✗ 启动失败: {result.stderr}")
            else:
                # sbatch 模式（原有逻辑）
                manager_script = generate_manager_script(config)
                manager_path = subs_dir / 'manager.slurm'

                with open(manager_path, 'w') as f:
                    f.write(manager_script)

                result = subprocess.run(
                    f"sbatch {manager_path}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=work_dir
                )

                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"✓ Manager 已提交到 SLURM")
                    print(f"  作业ID: {job_id}")
                    print(f"  日志: log/manager_{job_id}.log")
                else:
                    print(f"✗ 提交失败: {result.stderr}")

    elif args.action == 'cancel':
        _cancel_manager()
    else:
        print(f"未知操作: {args.action}")
        print("用法: qflow manager run|cancel")



def cmd_status(args):
    """显示任务状态 - 直接扫描文件系统"""
    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()
    structures_dir = (work_dir / config['manager']['structures_dir']).resolve()

    def get_task_status(task_path):
        """检查任务状态"""
        if (task_path / '.success').exists():
            return 'success'
        if (task_path / '.failed').exists():
            return 'failed'
        if (task_path / '.running').exists():
            return 'running'
        if (task_path / 'POSCAR').exists():
            return 'pending'
        return 'not_ready'

    # 收集所有任务
    all_tasks = {'opt': [], 'phonon': [], 'qha': [], 'bte': []}

    if structures_dir.exists():
        for struct_dir in structures_dir.iterdir():
            if not struct_dir.is_dir():
                continue

            # opt任务
            opt_dir = struct_dir / 'opt'
            if opt_dir.exists():
                status = get_task_status(opt_dir)
                if status != 'not_ready':
                    all_tasks['opt'].append({'path': str(opt_dir.relative_to(work_dir)), 'status': status})

            # phonon/qha任务
            for volume_dir in struct_dir.glob('volume_*'):
                for task_dir in volume_dir.glob('task.*'):
                    if not task_dir.is_dir() or task_dir.name == 'task_perfect':
                        continue

                    status = get_task_status(task_dir)
                    if status != 'not_ready':
                        task_type = 'qha' if volume_dir.name != 'volume_1.0' else 'phonon'
                        all_tasks[task_type].append({'path': str(task_dir.relative_to(work_dir)), 'status': status})

            # BTE 压强点任务 (P_XXGPa)
            for pressure_dir in struct_dir.glob('P_*GPa'):
                # 压强点优化任务
                bte_opt_dir = pressure_dir / 'opt'
                if bte_opt_dir.exists():
                    status = get_task_status(bte_opt_dir)
                    if status != 'not_ready':
                        all_tasks['bte'].append({'path': str(bte_opt_dir.relative_to(work_dir)), 'status': status})

                # BTE fc2/fc3 位移任务
                bte_dir = pressure_dir / 'bte'
                if bte_dir.exists():
                    for fc_type in ['fc2', 'fc3']:
                        fc_dir = bte_dir / fc_type
                        if fc_dir.exists():
                            for task_dir in fc_dir.glob('task.*'):
                                if not task_dir.is_dir() or task_dir.name == 'task_perfect':
                                    continue
                                status = get_task_status(task_dir)
                                if status != 'not_ready':
                                    all_tasks['bte'].append({'path': str(task_dir.relative_to(work_dir)), 'status': status})

    # 如果指定了 --running，显示正在运行的任务
    if hasattr(args, 'show_running') and args.show_running:
        running_tasks = {k: [t for t in v if t['status'] == 'running'] for k, v in all_tasks.items()}
        total_running = sum(len(v) for v in running_tasks.values())

        if total_running == 0:
            print("\n没有正在运行的任务")
            return

        # 读取 sbatch_jobs.json 构建 path -> job_id 反向映射
        jobs_file = work_dir / 'sbatch_jobs.json'
        path_to_jobid = {}
        if jobs_file.exists():
            try:
                job_mapping = json.loads(jobs_file.read_text())
                path_to_jobid = {v: k for k, v in job_mapping.items()}
            except Exception:
                pass

        # 从 squeue 获取所有 job 的实时状态（使用缩写：PD/R/CG 等）
        jobid_to_slurm_state = {}
        result = subprocess.run(
            "squeue -u $USER -o '%.18i %.2t' -h",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    jobid_to_slurm_state[parts[0].strip()] = parts[1].strip()

        # 从数据库读取运行时间
        from .task_db import TaskDB
        from datetime import datetime
        db = TaskDB(config)
        now = datetime.now()

        def get_scf_info(task_path: str) -> str:
            """读取OSZICAR或OUTCAR获取SCF信息"""
            task_dir = work_dir / task_path

            # 优先尝试OSZICAR
            oszicar = task_dir / 'OSZICAR'
            if oszicar.exists() and oszicar.stat().st_size > 0:
                try:
                    with open(oszicar, 'r') as f:
                        lines = f.readlines()
                    if lines:
                        ionic_steps = 0
                        last_e_step = 0
                        for line in lines:
                            line = line.strip()
                            if line.startswith(('DAV:', 'RMM:', 'CG:')):
                                last_e_step += 1
                            elif line and line[0].isdigit() and 'F=' in line:
                                ionic_steps += 1
                                last_e_step = 0
                        # 获取最后电子步的rms
                        rms = ""
                        for line in reversed(lines):
                            line = line.strip()
                            if line.startswith(('DAV:', 'RMM:', 'CG:')):
                                parts = line.split()
                                if len(parts) >= 5:
                                    rms = parts[-1]
                                break
                        return f"ion={ionic_steps} e={last_e_step} rms={rms}"
                except Exception:
                    pass

            # 回退到OUTCAR
            outcar = task_dir / 'OUTCAR'
            if outcar.exists():
                try:
                    import re
                    # 只读取最后20KB，避免读取大文件
                    file_size = outcar.stat().st_size
                    read_size = min(file_size, 20480)
                    with open(outcar, 'rb') as f:
                        if file_size > read_size:
                            f.seek(file_size - read_size)
                        content = f.read().decode('utf-8', errors='ignore')

                    # 从尾部找最后的 Iteration 行
                    # 格式: "------- Iteration      1(  14)  -------"
                    # 第一个数字是离子步，括号内是电子步
                    lines = content.split('\n')
                    ionic_step = 0
                    e_step = 0
                    for line in reversed(lines):
                        if 'Iteration' in line:
                            match = re.search(r'Iteration\s+(\d+)\(\s*(\d+)\)', line)
                            if match:
                                ionic_step = int(match.group(1))
                                e_step = int(match.group(2))
                                break

                    if ionic_step > 0 or e_step > 0:
                        return f"ion={ionic_step} e={e_step}"
                except Exception:
                    pass

            return ""

        # 收集所有running任务信息并按时间排序
        all_running = []
        for task_type in ['opt', 'phonon', 'qha', 'bte']:
            for task in running_tasks[task_type]:
                task_path = task['path']
                # 获取 job_id 和 SLURM 实时状态
                job_id = path_to_jobid.get(task_path, '-')
                slurm_state = jobid_to_slurm_state.get(job_id, '-')
                # 获取运行时间
                elapsed_min = 0
                for task_data in db.get_running_tasks():
                    if task_data['path'] == task_path:
                        try:
                            updated_at = datetime.fromisoformat(task_data.get('updated_at', ''))
                            elapsed_min = (now - updated_at).total_seconds() / 60
                        except:
                            pass
                        break
                # 获取SCF信息（优先OSZICAR，回退OUTCAR）
                scf_info = get_scf_info(task_path)
                all_running.append({
                    'path': task_path,
                    'type': task_type,
                    'elapsed': elapsed_min,
                    'scf_info': scf_info,
                    'job_id': job_id,
                    'slurm_state': slurm_state,
                })

        # 按运行时间排序（最长的在前）
        all_running.sort(key=lambda x: x['elapsed'], reverse=True)

        # 打印表头
        print(f"\n正在运行的任务 ({total_running} 个):")
        print("=" * 120)
        print(f"{'JobID':<10} {'ST':<5} {'Path':<55} {'Type':<8} {'Time':<10} {'SCF Info':<25}")
        print("-" * 120)

        for task in all_running:
            elapsed_str = f"{task['elapsed']:.1f} min" if task['elapsed'] > 0 else "-"
            path_display = task['path']
            if len(path_display) > 53:
                path_display = "..." + path_display[-50:]
            print(f"{task['job_id']:<10} {task['slurm_state']:<5} {path_display:<55} {task['type']:<8} {elapsed_str:<10} {task['scf_info']:<25}")

        return

    # 否则显示统计信息
    stats = {}
    failed_tasks_list = []

    for task_type, tasks in all_tasks.items():
        stats[task_type] = {'pending': 0, 'running': 0, 'success': 0, 'failed': 0}
        for task in tasks:
            status = task['status']
            if status in stats[task_type]:
                stats[task_type][status] += 1
            if status == 'failed':
                failed_tasks_list.append(task['path'])

    # 任务类型显示名称映射
    type_names = {
        'opt': 'Optimization',
        'phonon': 'Phonon',
        'qha': 'QHA',
        'bte': 'BTE'
    }

    total = {'pending': 0, 'running': 0, 'success': 0, 'failed': 0}

    # 收集数据
    rows = []
    for task_type in ['opt', 'phonon', 'qha', 'bte']:
        counts = stats.get(task_type, {'pending': 0, 'running': 0, 'success': 0, 'failed': 0})

        name = type_names.get(task_type, task_type)
        rows.append((name, counts['pending'], counts['running'], counts['success'], counts['failed']))

        for status in total:
            total[status] += counts.get(status, 0)

    # 打印表格
    print()
    print("┌──────────────┬─────────┬─────────┬─────────┬─────────┐")
    print("│ Stage        │ Pending │ Running │ Success │ Failed  │")
    print("├──────────────┼─────────┼─────────┼─────────┼─────────┤")

    for name, pending, running, success, failed in rows:
        print(f"│ {name:<12} │ {pending:>7} │ {running:>7} │ {success:>7} │ {failed:>7} │")

    print("├──────────────┼─────────┼─────────┼─────────┼─────────┤")
    print(f"│ {'Total':<12} │ {total['pending']:>7} │ {total['running']:>7} │ {total['success']:>7} │ {total['failed']:>7} │")
    print("└──────────────┴─────────┴─────────┴─────────┴─────────┘")

    # 统计虚频结构数量
    qha_dir = work_dir / 'qha_structures'
    if qha_dir.exists():
        imag_count = len(list(qha_dir.glob('*/.has_imag')))
        total_structs = len([d for d in qha_dir.iterdir() if d.is_dir()])
        if imag_count > 0:
            print(f"\n⚠ 存在虚频的结构: {imag_count}/{total_structs}")

    # 统计最近完成任务的平均执行时间
    from .task_db import TaskDB
    from datetime import datetime, timedelta
    db = TaskDB(config)
    now = datetime.now()
    hours_ago = now - timedelta(hours=6)  # 最近6小时

    # 读取最近完成的任务
    avg_times = {}
    for task_data in db.get_recent_completed(hours=6):
        try:
            task_type = task_data.get('task_type', 'unknown')
            if task_type not in avg_times:
                avg_times[task_type] = []
            avg_times[task_type].append(task_data['duration_seconds'])
        except:
            pass

    if avg_times:
        print("\n最近6小时平均执行时间:")
        remaining_time = 0
        for task_type, times in avg_times.items():
            avg_min = sum(times) / len(times) / 60
            pending_count = stats.get(task_type, {}).get('pending', 0)
            remaining_time += avg_min * pending_count
            print(f"  {task_type}: {avg_min:.1f} min (样本: {len(times)})")

        if remaining_time > 0 and total['running'] > 0:
            # 考虑并发数
            max_workers = 1
            max_workers_file = work_dir / 'max_workers.txt'
            if max_workers_file.exists():
                try:
                    max_workers = int(max_workers_file.read_text().strip())
                except:
                    pass
            estimated_hours = remaining_time / 60 / max_workers
            print(f"\n预计剩余时间: {estimated_hours:.1f} 小时 (并发: {max_workers})")

    # 显示失败任务（最多5个）
    if failed_tasks_list:
        print("\nRecent failed tasks:")
        for path in failed_tasks_list[:5]:
            print(f"  - {path}")

    print()


def cmd_cancel(args):
    """取消所有manager和任务"""
    cancelled = 0
    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()
    jobs_file = work_dir / 'sbatch_jobs.json'

    # 1. 取消 manager
    if _cancel_manager():
        cancelled += 1

    # 2. 仅取消当前工作目录下这个 manager 记录过的任务作业
    job_mapping = {}
    if jobs_file.exists():
        try:
            job_mapping = json.loads(jobs_file.read_text())
        except Exception:
            job_mapping = {}

    task_job_ids = []
    if job_mapping:
        result = subprocess.run(
            "squeue -u $USER -o '%.18i' -h",
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            active_job_ids = set(result.stdout.strip().split())
            task_job_ids = [job_id for job_id in job_mapping if job_id in active_job_ids]

        if task_job_ids:
            print(f"找到当前 manager 提交的 {len(task_job_ids)} 个任务作业")
            for job_id in task_job_ids:
                subprocess.run(f"scancel {job_id}", shell=True)
                cancelled += 1
            print(f"  已取消 {len(task_job_ids)} 个任务作业")
        else:
            print("没有找到当前 manager 提交且仍在运行的任务作业")
    else:
        print(f"没有找到当前工作目录的任务映射文件: {jobs_file}")

    if cancelled == 0:
        print("没有找到运行中的任务")
    else:
        print(f"\n总共取消了 {cancelled} 个作业")

    # 仅清理当前 manager 记录过的任务对应的 .running 标记
    print("\n清理当前 manager 任务的 .running 标记...")
    cleaned = 0
    for rel_path in job_mapping.values():
        task_dir = work_dir / rel_path
        running_file = task_dir / '.running'
        if running_file.exists():
            running_file.unlink()
            cleaned += 1

    if job_mapping:
        remaining_jobs = {job_id: path for job_id, path in job_mapping.items() if job_id not in task_job_ids}
        jobs_file.write_text(json.dumps(remaining_jobs, indent=2))

    if cleaned > 0:
        print(f"  已清理 {cleaned} 个 .running 标记")


def cmd_reset(args):
    """重置任务状态 - 处理文件系统标记和数据库状态"""
    from .task_db import TaskDB

    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()
    structures_dir = (work_dir / config['manager']['structures_dir']).resolve()

    # 检查是否有manager在运行
    running, info = _is_manager_running(config)
    if running:
        mode, _ = _get_manager_mode()
        if mode == 'local':
            print(f"注意: Manager正在运行 ({info})，reset将在线执行")
        else:
            print(f"错误: Manager正在远程运行 ({info})，请先执行 'qflow cancel' 停止Manager后再reset")
            return

    db = TaskDB(config)

    if args.running:
        # 清理所有.running标记
        count = 0
        if structures_dir.exists():
            for struct_dir in structures_dir.iterdir():
                if not struct_dir.is_dir():
                    continue

                # opt任务
                opt_dir = struct_dir / 'opt'
                if opt_dir.exists() and (opt_dir / '.running').exists():
                    (opt_dir / '.running').unlink()
                    count += 1

                # phonon/qha任务
                for volume_dir in struct_dir.glob('volume_*'):
                    for task_dir in volume_dir.glob('task.*'):
                        if task_dir.is_dir() and task_dir.name != 'task_perfect':
                            if (task_dir / '.running').exists():
                                (task_dir / '.running').unlink()
                                count += 1

                # bte压强点任务
                for pressure_dir in struct_dir.glob('P_*GPa'):
                    bte_opt_dir = pressure_dir / 'opt'
                    if bte_opt_dir.exists() and (bte_opt_dir / '.running').exists():
                        (bte_opt_dir / '.running').unlink()
                        count += 1

                    bte_dir = pressure_dir / 'bte'
                    if bte_dir.exists():
                        for fc_type in ['fc2', 'fc3']:
                            fc_dir = bte_dir / fc_type
                            if fc_dir.exists():
                                for task_dir in fc_dir.glob('task.*'):
                                    if task_dir.is_dir() and task_dir.name != 'task_perfect':
                                        if (task_dir / '.running').exists():
                                            (task_dir / '.running').unlink()
                                            count += 1

        # 更新数据库
        db_count = db.reset_running_tasks()

        print(f"已清理 {count} 个 .running 标记")
        print(f"数据库更新: {db_count} 条记录")

    elif args.failed:
        # 清理所有.failed标记
        count = 0
        if structures_dir.exists():
            for struct_dir in structures_dir.iterdir():
                if not struct_dir.is_dir():
                    continue

                # opt任务
                opt_dir = struct_dir / 'opt'
                if opt_dir.exists() and (opt_dir / '.failed').exists():
                    (opt_dir / '.failed').unlink()
                    error_log = opt_dir / 'error.log'
                    if error_log.exists():
                        error_log.unlink()
                    count += 1

                # phonon/qha任务
                for volume_dir in struct_dir.glob('volume_*'):
                    for task_dir in volume_dir.glob('task.*'):
                        if task_dir.is_dir() and task_dir.name != 'task_perfect':
                            if (task_dir / '.failed').exists():
                                (task_dir / '.failed').unlink()
                                error_log = task_dir / 'error.log'
                                if error_log.exists():
                                    error_log.unlink()
                                count += 1

                # bte压强点任务
                for pressure_dir in struct_dir.glob('P_*GPa'):
                    bte_opt_dir = pressure_dir / 'opt'
                    if bte_opt_dir.exists() and (bte_opt_dir / '.failed').exists():
                        (bte_opt_dir / '.failed').unlink()
                        error_log = bte_opt_dir / 'error.log'
                        if error_log.exists():
                            error_log.unlink()
                        count += 1

                    bte_dir = pressure_dir / 'bte'
                    if bte_dir.exists():
                        for fc_type in ['fc2', 'fc3']:
                            fc_dir = bte_dir / fc_type
                            if fc_dir.exists():
                                for task_dir in fc_dir.glob('task.*'):
                                    if task_dir.is_dir() and task_dir.name != 'task_perfect':
                                        if (task_dir / '.failed').exists():
                                            (task_dir / '.failed').unlink()
                                            error_log = task_dir / 'error.log'
                                            if error_log.exists():
                                                error_log.unlink()
                                            count += 1

        # 更新数据库
        db_count = db.reset_failed_tasks()

        print(f"已清理 {count} 个 .failed 标记")
        print(f"数据库更新: {db_count} 条记录")
    else:
        print("用法:")
        print("  qflow reset --running    # 重置running任务为pending")
        print("  qflow reset --failed     # 重置failed任务为pending")


def cmd_regen(args):
    """按任务类别重新生成VASP输入文件并重置为pending"""
    import subprocess
    import shutil
    from tqdm import tqdm
    from .task_db import TaskDB
    from pymatgen.io.vasp.sets import MPRelaxSet, MatPESStaticSet
    from pymatgen.core import Structure as PMGStructure

    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()
    structures_dir = (work_dir / config['manager']['structures_dir']).resolve()

    # 检查是否有manager在运行
    running, info = _is_manager_running(config)
    if running:
        print(f"错误: Manager正在运行 ({info})，请先执行 'qflow cancel' 停止Manager后再regen")
        return

    db = TaskDB(config)

    # 获取INCAR设置
    incar_config = config.get('incar', {})
    incar_settings_opt = incar_config.get('opt', {})
    incar_settings_phonon = incar_config.get('phonon', {})

    # 获取POTCAR设置
    potcar_config = config.get('potcar', {})
    potcar_functional = potcar_config.get('functional', 'PBE_54')

    task_type = args.type
    print(f"重新生成 {task_type} 类型任务的VASP输入文件...")

    regen_count = 0
    reset_count = 0
    cascade_deleted = 0

    if not structures_dir.exists():
        print(f"错误: 结构目录不存在 {structures_dir}")
        return

    # 收集所有结构目录
    struct_dirs = [d for d in structures_dir.iterdir() if d.is_dir()]

    for struct_dir in tqdm(struct_dirs, desc=f"Regen {task_type}", unit="struct"):

        if task_type == 'opt':
            # 处理opt任务
            opt_dir = struct_dir / 'opt'
            if opt_dir.exists():
                poscar = opt_dir / 'POSCAR'
                if poscar.exists():
                    try:
                        # 清理状态标记
                        for marker in ['.running', '.completed', '.failed', '.success']:
                            m = opt_dir / marker
                            if m.exists():
                                m.unlink()

                        structure = PMGStructure.from_file(str(poscar))
                        mp_set = MPRelaxSet(structure, user_incar_settings=incar_settings_opt,
                                           user_potcar_functional=potcar_functional)
                        mp_set.write_input(str(opt_dir))
                        regen_count += 1

                        # 更新数据库
                        task_path = str(opt_dir.relative_to(work_dir))
                        db.reset_task_to_pending(task_path)
                        reset_count += 1

                        # 级联删除: 删除所有volume_*目录
                        for volume_dir in struct_dir.glob('volume_*'):
                            # 删除数据库中的任务
                            for task_dir in volume_dir.glob('task.*'):
                                if task_dir.is_dir() and task_dir.name != 'task_perfect':
                                    task_path = str(task_dir.relative_to(work_dir))
                                    db.remove_task(task_path)
                                    cascade_deleted += 1
                            # 删除目录
                            shutil.rmtree(volume_dir)
                        # 清理.has_imag标记
                        has_imag = struct_dir / '.has_imag'
                        if has_imag.exists():
                            has_imag.unlink()

                    except Exception as e:
                        print(f"  警告: {opt_dir} - {e}")

        elif task_type == 'phonon':
            # 检查opt是否完成
            opt_dir = struct_dir / 'opt'
            if not (opt_dir / '.success').exists():
                # opt未完成，跳过此结构
                continue

            # 处理phonon任务 (volume_1.0)
            volume_dir = struct_dir / 'volume_1.0'

            # 如果volume_1.0不存在，创建新的phonon任务
            if not volume_dir.exists():
                from ase.io import read, write
                from .phonon_utils import generate_phonon_displacements

                # 读取优化后的结构
                contcar = opt_dir / 'CONTCAR'
                if contcar.exists() and contcar.stat().st_size > 0:
                    atoms = read(str(contcar))
                else:
                    poscar = opt_dir / 'POSCAR'
                    if poscar.exists():
                        atoms = read(str(poscar))
                    else:
                        continue

                # 获取phonon配置
                phonon_config = config.get('phonon', {})
                supercell = phonon_config.get('supercell', None)
                max_atoms = phonon_config.get('max_atoms', None)
                min_atoms = phonon_config.get('min_atoms', 100)
                min_length = phonon_config.get('min_length', 10.0)
                distance = phonon_config.get('displacement_distance', 0.01)

                try:
                    n_tasks = generate_phonon_displacements(
                        atoms=atoms,
                        volume_dir=str(volume_dir),
                        supercell=supercell,
                        max_atoms=max_atoms,
                        min_atoms=min_atoms,
                        min_length=min_length,
                        distance=distance
                    )

                    # 生成VASP输入并添加到数据库
                    for i in range(n_tasks):
                        task_dir = volume_dir / f'task.{i:06d}'
                        poscar = task_dir / 'POSCAR'
                        if poscar.exists():
                            structure = PMGStructure.from_file(str(poscar))
                            mp_set = MatPESStaticSet(structure, user_incar_settings=incar_settings_phonon,
                                                user_potcar_functional=potcar_functional)
                            mp_set.write_input(str(task_dir))
                            regen_count += 1

                            task_path = str(task_dir.relative_to(work_dir))
                            db.add_task(task_path, 'phonon')
                            reset_count += 1

                    # 清理.phonon_done标记（需要重新后处理）
                    phonon_done = struct_dir / '.phonon_done'
                    if phonon_done.exists():
                        phonon_done.unlink()

                except Exception as e:
                    print(f"  警告: 创建phonon任务失败 {struct_dir.name} - {e}")
            else:
                # volume_1.0存在，重新生成已有任务的输入文件
                for task_dir in volume_dir.glob('task.*'):
                    if task_dir.is_dir() and task_dir.name != 'task_perfect':
                        poscar = task_dir / 'POSCAR'
                        if poscar.exists():
                            try:
                                # 清理状态标记
                                for marker in ['.running', '.completed', '.failed', '.success']:
                                    m = task_dir / marker
                                    if m.exists():
                                        m.unlink()

                                structure = PMGStructure.from_file(str(poscar))
                                mp_set = MatPESStaticSet(structure, user_incar_settings=incar_settings_phonon,
                                                    user_potcar_functional=potcar_functional)
                                mp_set.write_input(str(task_dir))
                                regen_count += 1

                                # 更新数据库
                                task_path = str(task_dir.relative_to(work_dir))
                                db.reset_task_to_pending(task_path)
                                reset_count += 1
                            except Exception as e:
                                print(f"  警告: {task_dir} - {e}")

                # 清理.phonon_done标记（需要重新后处理）
                phonon_done = struct_dir / '.phonon_done'
                if phonon_done.exists():
                    phonon_done.unlink()

            # 级联删除: 删除qha的volume_*目录(除了volume_1.0)
            for qha_volume_dir in struct_dir.glob('volume_*'):
                if qha_volume_dir.name == 'volume_1.0':
                    continue
                # 删除数据库中的任务
                for task_dir in qha_volume_dir.glob('task.*'):
                    if task_dir.is_dir() and task_dir.name != 'task_perfect':
                        task_path = str(task_dir.relative_to(work_dir))
                        db.remove_task(task_path)
                        cascade_deleted += 1
                # 删除目录
                shutil.rmtree(qha_volume_dir)
            # 清理.has_imag标记
            has_imag = struct_dir / '.has_imag'
            if has_imag.exists():
                has_imag.unlink()

        elif task_type == 'qha':
            # 检查opt和phonon是否完成
            opt_dir = struct_dir / 'opt'
            if not (opt_dir / '.success').exists():
                continue
            # 检查phonon是否完成
            if not (struct_dir / '.phonon_done').exists():
                continue
            # 检查是否有虚频（有虚频的不应该做qha）
            if (struct_dir / '.has_imag').exists():
                continue

            # 处理qha任务 (volume_* 除了 volume_1.0)
            for volume_dir in struct_dir.glob('volume_*'):
                if volume_dir.name == 'volume_1.0':
                    continue
                for task_dir in volume_dir.glob('task.*'):
                    if task_dir.is_dir() and task_dir.name != 'task_perfect':
                        poscar = task_dir / 'POSCAR'
                        if poscar.exists():
                            try:
                                # 清理状态标记
                                for marker in ['.running', '.completed', '.failed', '.success']:
                                    m = task_dir / marker
                                    if m.exists():
                                        m.unlink()

                                structure = PMGStructure.from_file(str(poscar))
                                mp_set = MatPESStaticSet(structure, user_incar_settings=incar_settings_phonon,
                                                    user_potcar_functional=potcar_functional)
                                mp_set.write_input(str(task_dir))
                                regen_count += 1

                                # 更新数据库
                                task_path = str(task_dir.relative_to(work_dir))
                                db.reset_task_to_pending(task_path)
                                reset_count += 1
                            except Exception as e:
                                print(f"  警告: {task_dir} - {e}")

    print(f"已重新生成 {regen_count} 个任务的VASP输入文件")
    print(f"已重置 {reset_count} 个任务为pending状态")
    if cascade_deleted > 0:
        print(f"已级联删除 {cascade_deleted} 个下游任务")


def cmd_sync(args):
    """同步任务队列（扫描文件系统并更新队列状态）"""
    print("=== QFlow 队列同步 ===\n")

    config = load_config()
    running, info = _is_manager_running(config)

    if running:
        # manager 在运行，先停掉再重启
        print(f"检测到运行中的 Manager ({info})")
        print("停止 Manager...")
        _cancel_manager(config)
        print()
        time.sleep(2)

        # 重新启动 manager（启动时会自动同步）
        print("重新启动 Manager...")
        work_dir = Path(config.get('work_dir', '.')).resolve()
        subprocess.run(
            [sys.executable, '-m', 'qflow.cli', 'manager', 'run'],
            cwd=work_dir
        )
    else:
        # manager 没有运行，直接同步
        print("Manager未运行，执行队列同步...\n")

        from .manager import Manager

        manager = Manager(config)
        synced_counts = manager.sync_queue_from_filesystem()

        # 显示同步结果
        print("\n同步完成！")
        print(f"  新增任务: {synced_counts['added']}")
        print(f"  更新为success: {synced_counts['updated_success']}")
        print(f"  更新为failed: {synced_counts['updated_failed']}")
        print(f"  更新为running: {synced_counts['updated_running']}")
        print(f"  删除过期任务: {synced_counts['removed']}")

        # 显示队列统计
        stats = manager.db.get_statistics()
        print(f"\n当前数据库状态:")
        for task_type, counts in stats.items():
            print(f"  {task_type}: pending={counts['pending']}, running={counts['running']}, "
                  f"success={counts['success']}, failed={counts['failed']}")

        print("\n提示: 使用 'qflow manager run' 启动 Manager")


def cmd_prepare(args):
    """生成SLURM脚本"""
    config = load_config()
    work_dir = Path(config.get('work_dir', '.')).resolve()

    # 创建目录
    subs_dir = work_dir / 'subs'
    log_dir = work_dir / 'log'
    subs_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # 生成 manager 脚本
    manager_script = f"""#!/bin/bash
#SBATCH --job-name=qflow_manager
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={args.manager_cores}
#SBATCH --partition=cpu
#SBATCH --time={args.manager_time}
#SBATCH --output=log/manager_%j.log
#SBATCH --error=log/manager_%j.log

# 激活 conda 环境
source {Path.home()}/miniforge3/etc/profile.d/conda.sh
conda activate mattersim

cd {work_dir}

qflow manager run
"""

    manager_path = subs_dir / 'manager.slurm'
    with open(manager_path, 'w') as f:
        f.write(manager_script)

    print(f"✓ 已生成 Manager 脚本: {manager_path}")
    print(f"  核心数: {args.manager_cores}")
    print(f"  时间限制: {args.manager_time}")
    print()
    print("使用方法:")
    print(f"  sbatch {manager_path}           # 提交 manager")
    print(f"  qflow worker 25                 # 提交 25 个 workers")


def main():
    parser = argparse.ArgumentParser(
        prog='qflow',
        description='计算任务工作流管理系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  qflow manager run                    # 启动Manager进程
  qflow manager cancel                 # 取消Manager作业
  qflow worker run -n 3 -g "2,3"       # 启动3个worker，使用GPU 2,3
  qflow worker 10                      # 提交10个worker到SLURM
  qflow worker cancel                  # 停止所有worker
  qflow status                         # 查看任务进度
  qflow sync                           # 同步任务队列
  qflow cancel                         # 取消所有manager和worker
  qflow reset --running                # 重置running任务为pending
'''
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # worker子命令
    parser_worker = subparsers.add_parser(
        'worker',
        help='Worker管理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='管理Worker进程',
        epilog='''
操作类型:
  run       直接启动Python worker进程（推荐测试使用）
  cancel    停止所有通过run启动的worker
  <N>       提交N个worker作业到SLURM（生产环境）

示例:
  qflow worker run -n 3 -g "2,3" --max-idle 180   # 启动3个worker
  qflow worker run --mode vasp                     # 使用VASP模式
  qflow worker 50                                  # 提交50个SLURM作业
  qflow worker cancel                              # 停止所有worker
'''
    )
    parser_worker.add_argument('action', help='操作: run/cancel/<N>')
    parser_worker.add_argument('-n', '--num', type=int, default=1,
                               help='worker数量 (默认: 1)')
    parser_worker.add_argument('-g', '--gpu', type=str,
                               help='GPU列表，如 "2,3"，多worker轮流分配')
    parser_worker.add_argument('--max-idle', type=int, default=120,
                               help='最大空闲时间(秒)，超时自动退出 (默认: 120)')
    parser_worker.add_argument('--mode', type=str, choices=['mattersim', 'vasp'],
                               help='计算模式，覆盖config.yaml设置')
    parser_worker.set_defaults(func=cmd_worker)

    # manager子命令
    parser_manager = subparsers.add_parser(
        'manager',
        help='Manager管理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='管理Manager进程，负责任务生成和后处理',
        epilog='''
操作类型:
  run       提交manager作业到SLURM
  cancel    取消manager作业

示例:
  qflow manager run       # 提交manager到SLURM
  qflow manager cancel    # 取消manager作业
'''
    )
    parser_manager.add_argument('action', help='操作: run/cancel')
    parser_manager.set_defaults(func=cmd_manager)

    # status子命令
    parser_status = subparsers.add_parser(
        'status',
        help='查看任务状态',
        description='显示各阶段任务的执行进度统计'
    )
    parser_status.add_argument(
        '--running',
        dest='show_running',
        action='store_true',
        help='显示正在运行的任务详情'
    )
    parser_status.set_defaults(func=cmd_status)

    # cancel子命令
    parser_cancel = subparsers.add_parser(
        'cancel',
        help='取消所有任务',
        description='取消所有运行中的manager和worker'
    )
    parser_cancel.set_defaults(func=cmd_cancel)

    # reset子命令
    parser_reset = subparsers.add_parser(
        'reset',
        help='重置任务状态',
        description='重置任务状态，用于故障恢复'
    )
    parser_reset.add_argument('--running', action='store_true',
                              help='将所有running状态的任务重置为pending')
    parser_reset.add_argument('--failed', action='store_true',
                              help='将所有failed状态的任务重置为pending（同时删除.failed文件）')
    parser_reset.set_defaults(func=cmd_reset)

    # regen子命令
    parser_regen = subparsers.add_parser(
        'regen',
        help='重新生成VASP输入文件',
        description='按任务类别重新生成VASP输入文件并重置为pending'
    )
    parser_regen.add_argument('type', choices=['opt', 'phonon', 'qha'],
                              help='任务类型: opt(优化), phonon(声子), qha(QHA)')
    parser_regen.set_defaults(func=cmd_regen)

    # sync子命令
    parser_sync = subparsers.add_parser(
        'sync',
        help='同步任务队列',
        description='扫描文件系统，删除不存在的任务，添加新的任务'
    )
    parser_sync.set_defaults(func=cmd_sync)

    # prepare子命令
    parser_prepare = subparsers.add_parser(
        'prepare',
        help='生成SLURM脚本',
        description='自动生成manager和worker的SLURM提交脚本'
    )
    parser_prepare.add_argument('--manager-cores', type=int, default=16,
                               help='Manager使用的核心数 (默认: 16)')
    parser_prepare.add_argument('--manager-time', type=str, default='150:00:00',
                               help='Manager时间限制 (默认: 150:00:00)')
    parser_prepare.set_defaults(func=cmd_prepare)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
