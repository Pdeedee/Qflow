#!/usr/bin/env python3
"""SLURM 脚本模板生成"""

import sys
import os
from pathlib import Path


def generate_worker_script(config: dict) -> str:
    """根据配置生成 Worker SLURM 脚本内容"""
    slurm_config = config.get('slurm', {})
    work_dir = Path(config.get('work_dir', '.')).resolve()

    lines = ["#!/bin/bash"]

    # SBATCH header参数映射
    sbatch_params = {
        'nodes': 'nodes',
        'ntasks_per_node': 'ntasks-per-node',
        'partition': 'partition',
        'time': 'time',
        'mem': 'mem',
        'gres': 'gres',
        'account': 'account',
        'qos': 'qos',
        'constraint': 'constraint',
        'exclude': 'exclude',
    }

    # 生成SBATCH header
    for config_key, sbatch_key in sbatch_params.items():
        value = slurm_config.get(config_key)
        if value is not None:
            lines.append(f"#SBATCH --{sbatch_key}={value}")

    # 输出文件（合并到同一个 log 文件）
    lines.append("#SBATCH --output=log/%j.log")
    lines.append("#SBATCH --error=log/%j.log")
    lines.append("")

    # 获取 Python 路径
    python_path = slurm_config.get('python_path')
    if not python_path:
        # 如果没有指定，使用当前 Python 解释器路径
        python_path = sys.executable

    # 加载模块（如果需要）
    modules = slurm_config.get('modules', [])
    if modules:
        # 获取 module 初始化脚本路径
        module_init = slurm_config.get('module_init_script')
        if module_init:
            lines.append("# 初始化 module 系统")
            lines.append(f"if [ -f {module_init} ]; then")
            lines.append(f"    source {module_init}")
            lines.append("fi")
            lines.append("")

        lines.append("# 加载模块")
        for module in modules:
            lines.append(f"module load {module} 2>/dev/null || true")
        lines.append("")

    # 额外命令
    extra_commands = slurm_config.get('extra_commands', [])
    if extra_commands:
        for cmd in extra_commands:
            lines.append(cmd)
        lines.append("")

    # 设置工作目录
    lines.append(f"cd {work_dir}")
    lines.append("")

    # 运行worker（使用绝对路径）
    lines.append(f"# 使用 Python 绝对路径运行 worker")
    lines.append(f"{python_path} -m qflow.worker")
    lines.append("")

    return "\n".join(lines)


def generate_manager_script(config: dict) -> str:
    """生成 Manager SLURM 脚本"""
    slurm_config = config.get('slurm', {})
    work_dir = Path(config.get('work_dir', '.')).resolve()

    # 获取 manager 配置（如果没有就用默认值）
    manager_cores = 16
    manager_time = '150:00:00'
    partition = slurm_config.get('partition', 'cpu')

    # 获取 Python 路径
    python_path = slurm_config.get('python_path')
    if not python_path:
        # 如果没有指定，使用当前 Python 解释器路径
        python_path = sys.executable

    script = f"""#!/bin/bash
#SBATCH --job-name=qflow_manager
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={manager_cores}
#SBATCH --partition={partition}
#SBATCH --time={manager_time}
#SBATCH --output=log/manager_%j.log
#SBATCH --error=log/manager_%j.log

cd {work_dir}

# 使用 Python 绝对路径运行 manager
{python_path} -m qflow.manager
"""
    return script


def generate_task_script(config: dict, task_name: str = "qflow_task") -> str:
    """生成任务执行的SLURM脚本（在任务目录中执行）

    Args:
        config: 配置字典
        task_name: 任务名称（用于SBATCH --job-name）

    Returns:
        SLURM脚本内容
    """
    slurm_config = config.get('slurm', {})
    worker_config = config.get('worker', {})

    # 获取基本配置
    nodes = slurm_config.get('nodes', 1)
    ntasks = slurm_config.get('ntasks_per_node', 36)
    partition = slurm_config.get('partition', 'cpu')
    time_limit = slurm_config.get('task_time', '24:00:00')  # 单个任务默认1天
    vasp_cmd = worker_config.get('vasp_cmd', 'mpirun vasp_std')

    # timeout设置为SLURM时间限制的90%（留余量给清理工作）
    # 解析time_limit为秒数
    time_parts = time_limit.split(':')
    if len(time_parts) == 3:
        hours, mins, secs = map(int, time_parts)
        timeout = int((hours * 3600 + mins * 60 + secs) * 0.95)
    else:
        timeout = 82800  # 默认23小时

    # 可选的SBATCH参数
    optional_params = ""
    for key, sbatch_key in [('mem', 'mem'), ('gres', 'gres'), ('account', 'account'),
                            ('qos', 'qos'), ('constraint', 'constraint'), ('exclude', 'exclude'),
                            ('nodelist', 'nodelist')]:
        value = slurm_config.get(key)
        if value:
            optional_params += f"#SBATCH --{sbatch_key}={value}\n"

    # 模块加载
    module_section = ""
    modules = slurm_config.get('modules', [])
    if modules:
        module_init = slurm_config.get('module_init_script')
        if module_init:
            module_section += f"""# 初始化 module 系统
if [ -f {module_init} ]; then
    source {module_init}
fi

"""
        module_section += "# 加载模块\n"
        for module in modules:
            module_section += f"module load {module} 2>/dev/null || true\n"
        module_section += "\n"

    # 额外环境设置
    env_section = ""
    extra_commands = slurm_config.get('extra_commands', [])
    if extra_commands:
        env_section = "# 环境设置\n"
        for cmd in extra_commands:
            env_section += f"{cmd}\n"
        env_section += "\n"

    script = f"""#!/bin/bash
#SBATCH --job-name={task_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
{optional_params}#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.log

{module_section}{env_section}# 任务执行
echo "=========================================="
echo "QFlow Task"
echo "=========================================="
echo "Task directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_ISO=$(date -Iseconds)
echo "$START_TIME_ISO" > .start_time

# 捕获SIGTERM信号（SLURM取消/超时/节点故障时触发）
cleanup() {{
    echo "SIGNAL received, marking task as failed at $(date)"
    echo "Task killed by signal at $(date)" > error.log
    tail -100 vasp.log >> error.log 2>/dev/null || true
    touch .failed
    rm -f .running
    exit 1
}}
trap cleanup SIGTERM SIGINT SIGHUP

# 执行VASP
echo "Executing VASP with timeout {timeout}s..."
echo "Command: {vasp_cmd}"

# 执行并捕获退出码
set +e
timeout {timeout} bash -c '{vasp_cmd}' > vasp.log 2>&1
EXIT_CODE=$?
set -e

# 记录结束时间
END_TIME=$(date +%s)
END_TIME_ISO=$(date -Iseconds)
DURATION=$((END_TIME - START_TIME))

echo "VASP finished with exit code: $EXIT_CODE"
echo "Duration: $DURATION seconds"

# 写入任务时间记录
cat > .task_time << EOF
start_time: $START_TIME_ISO
end_time: $END_TIME_ISO
duration_seconds: $DURATION
exit_code: $EXIT_CODE
EOF

# 设置任务状态
if [ $EXIT_CODE -eq 124 ]; then
    # timeout退出码124表示超时
    echo "ERROR: VASP execution timeout"
    echo "VASP execution timeout after {timeout}s at $(date)" > error.log
    echo "status: timeout" >> .task_time
    touch .failed
    exit 1
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: VASP failed with exit code $EXIT_CODE"
    echo "VASP failed with exit code $EXIT_CODE at $(date)" > error.log
    tail -100 vasp.log >> error.log 2>/dev/null || true
    echo "status: failed" >> .task_time
    touch .failed
    exit 1
else
    echo "SUCCESS: VASP completed successfully"
    echo "status: success" >> .task_time
    touch .success
    exit 0
fi
"""
    return script
