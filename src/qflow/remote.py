# -*- coding: utf-8 -*-
"""Remote SLURM submission support for QFlow."""

import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import paramiko

from .logger import logger


def _run_cmd(cmd: List[str]):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        out, err = proc.communicate()
        return proc.returncode, out, err


def rsync(
    from_path: str,
    to_path: str,
    port: int = 22,
    key_filename: Optional[str] = None,
    timeout: int = 20,
    option_args: Optional[List[str]] = None,
    additional_args: Optional[List[str]] = None,
):
    ssh_cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        "-q",
    ]
    if key_filename:
        ssh_cmd.extend(["-i", key_filename])

    cmd = ["rsync", "-az", "-e", " ".join(ssh_cmd), "-q"]
    if option_args:
        cmd.extend(option_args)
    cmd.extend([from_path, to_path])
    if additional_args:
        cmd.extend(additional_args)

    ret, out, err = _run_cmd(cmd)
    if ret != 0:
        raise RuntimeError(f"rsync failed: {' '.join(cmd)}\n{err.decode('utf-8', errors='replace')}")


class RemoteRunner:
    """Submit and synchronize QFlow tasks on a remote SLURM cluster."""

    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("work_dir", ".")).resolve()
        self.remote_config = config.get("remote", {}) or {}
        self.hostname = self.remote_config.get("ssh_hostname")
        self.username = self.remote_config.get("ssh_username")
        self.port = int(self.remote_config.get("ssh_port", 22))
        self.timeout = int(self.remote_config.get("ssh_timeout", 20))
        self.remote_root = str(self.remote_config.get("remote_root", "")).rstrip("/")
        self.project_name = self.remote_config.get("project_name") or self.work_dir.name
        self.remote_work_dir = f"{self.remote_root}/{self.project_name}".rstrip("/")
        self.key_filename = self.remote_config.get("ssh_key_filename")
        self.ssh = None
        self._sftp = None
        self.remotename = f"{self.username}@{self.hostname}"

        if not self.hostname or not self.username or not self.remote_root:
            raise ValueError("remote mode requires remote.ssh_hostname, remote.ssh_username, and remote.remote_root")

        self._setup_ssh()

    def _setup_ssh(self):
        key_candidates = self._ssh_key_candidates()
        if not key_candidates:
            key_candidates = [None]

        errors = []
        for key_filename in key_candidates:
            key_msg = key_filename or "default SSH keys"
            for attempt in range(1, 4):
                self.ssh = paramiko.SSHClient()
                self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                logger.info(
                    f"remote ssh connecting to {self.hostname}:{self.port} as {self.username} "
                    f"with {key_msg} (attempt {attempt}/3)"
                )
                try:
                    self.ssh.connect(
                        hostname=self.hostname,
                        port=self.port,
                        username=self.username,
                        key_filename=key_filename,
                        timeout=self.timeout,
                        compress=True,
                        allow_agent=False,
                        look_for_keys=key_filename is None,
                    )
                except Exception as exc:
                    errors.append(f"{key_msg}: {type(exc).__name__}: {exc!r}")
                    logger.warning(f"remote ssh key failed: {key_msg}: {type(exc).__name__}: {exc!r}")
                    self.ssh.close()
                    time.sleep(2)
                    continue
                self.key_filename = key_filename
                logger.info("remote ssh connection established")
                return

        raise RuntimeError("all remote ssh keys failed:\n" + "\n".join(errors))

    def _ssh_key_candidates(self) -> List[str]:
        keys: List[str] = []

        def add_key(path):
            if not path:
                return
            path = os.path.expandvars(os.path.expanduser(str(path)))
            if os.path.isfile(path) and path not in keys:
                keys.append(path)

        raw_key = self.key_filename
        if isinstance(raw_key, (list, tuple)):
            for path in raw_key:
                add_key(path)
        else:
            add_key(raw_key)

        self._add_ssh_config_identity_files(keys)

        ssh_dir = Path.home() / ".ssh"
        if ssh_dir.is_dir():
            ignored = {"authorized_keys", "config", "known_hosts", "known_hosts2"}
            for path in sorted(ssh_dir.iterdir()):
                if not path.is_file():
                    continue
                name = path.name
                if name.startswith(".") or name.endswith(".pub") or name in ignored:
                    continue
                add_key(str(path))
        return keys

    def _add_ssh_config_identity_files(self, keys: List[str]):
        config_file = Path.home() / ".ssh" / "config"
        if not config_file.is_file():
            return
        try:
            with config_file.open() as fp:
                ssh_config = paramiko.SSHConfig()
                ssh_config.parse(fp)
            host_config = ssh_config.lookup(self.hostname)
        except Exception as exc:
            logger.warning(f"failed to parse SSH config {config_file}: {exc!r}")
            return
        for path in host_config.get("identityfile", []):
            path = os.path.expandvars(os.path.expanduser(str(path)))
            if os.path.isfile(path) and path not in keys:
                keys.append(path)

    def close(self):
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self.ssh is not None:
            self.ssh.close()
            self.ssh = None

    def ensure_alive(self):
        try:
            transport = self.ssh.get_transport() if self.ssh is not None else None
            if transport is None:
                raise EOFError
            transport.send_ignore()
        except Exception:
            self.close()
            self._setup_ssh()

    def _reconnect_ssh(self):
        self.close()
        self._setup_ssh()

    def block_call(self, cmd: str):
        remote_cmd = f"cd {shlex.quote(self.remote_root)} ; " + cmd
        connect_errors = (
            EOFError,
            OSError,
            ConnectionError,
            socket.error,
            paramiko.SSHException,
        )
        last_exc = None
        for attempt in range(1, 4):
            self.ensure_alive()
            try:
                stdin, stdout, stderr = self.ssh.exec_command(remote_cmd)
                break
            except connect_errors as exc:
                last_exc = exc
                logger.warning(
                    f"remote command session failed, reconnecting "
                    f"(attempt {attempt}/3): {type(exc).__name__}: {exc}"
                )
                self._reconnect_ssh()
                time.sleep(attempt)
        else:
            raise RuntimeError(f"remote command session failed after reconnects: {cmd}") from last_exc
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdout.read().decode("utf-8"), stderr.read().decode("utf-8")

    def block_checkcall(self, cmd: str):
        ret, out, err = self.block_call(cmd)
        if ret != 0:
            raise RuntimeError(f"remote command failed ({ret}): {cmd}\n{err}")
        return out

    def remote_task_path(self, task_path: str) -> str:
        return f"{self.remote_work_dir}/{task_path}"

    def rsync_send_files(self, files: Iterable[str]):
        files = self._expand_files_from(files)
        if not files:
            return
        self.block_checkcall(f"mkdir -p {shlex.quote(self.remote_work_dir)}")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            files_from = fp.name
            for path in files:
                fp.write(path.rstrip("/") + "\n")
        try:
            rsync(
                str(self.work_dir) + "/",
                f"{self.remotename}:{self.remote_work_dir}/",
                port=self.port,
                key_filename=self.key_filename,
                timeout=self.timeout,
                option_args=[f"--files-from={files_from}"],
            )
        finally:
            try:
                os.remove(files_from)
            except OSError:
                pass

    def _expand_files_from(self, paths: Iterable[str]) -> List[str]:
        expanded = []
        for raw_path in paths:
            raw_path = str(raw_path).rstrip("/")
            local_path = self.work_dir / raw_path
            if local_path.is_dir():
                for path in sorted(local_path.rglob("*")):
                    if path.is_file():
                        expanded.append(path.relative_to(self.work_dir).as_posix())
            elif local_path.exists():
                expanded.append(local_path.relative_to(self.work_dir).as_posix())
            else:
                expanded.append(raw_path)
        return list(dict.fromkeys(expanded))

    def rsync_fetch_task(self, task_path: str):
        local_path = self.work_dir / task_path
        local_path.mkdir(parents=True, exist_ok=True)
        remote_path = self.remote_task_path(task_path).rstrip("/") + "/"
        rsync(
            f"{self.remotename}:{remote_path}",
            str(local_path) + "/",
            port=self.port,
            key_filename=self.key_filename,
            timeout=self.timeout,
            additional_args=["--exclude=POTCAR", "--exclude=*.xml"],
        )

    def rsync_fetch_tasks(self, task_paths: Iterable[str]):
        task_paths = [str(path).rstrip("/") for path in task_paths]
        if not task_paths:
            return

        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            files_from = fp.name
            for task_path in task_paths:
                (self.work_dir / task_path).mkdir(parents=True, exist_ok=True)
                fp.write(task_path + "/\n")
        try:
            rsync(
                f"{self.remotename}:{self.remote_work_dir}/",
                str(self.work_dir) + "/",
                port=self.port,
                key_filename=self.key_filename,
                timeout=self.timeout,
                option_args=[f"--files-from={files_from}", "--relative"],
                additional_args=["--exclude=POTCAR", "--exclude=*.xml"],
            )
        finally:
            try:
                os.remove(files_from)
            except OSError:
                pass

    def remove_remote_tasks(self, task_paths: Iterable[str]):
        task_paths = [str(path).strip("/") for path in task_paths if str(path).strip("/")]
        if not task_paths:
            return
        quoted_paths = " ".join(
            shlex.quote(f"{self.remote_work_dir}/{task_path}") for task_path in task_paths
        )
        self.block_checkcall(f"rm -rf -- {quoted_paths}")

    def submit_task(self, task_path: str) -> Optional[str]:
        remote_task = self.remote_task_path(task_path)
        cmd = f"cd {shlex.quote(remote_task)} && sbatch run.sbatch"
        ret, out, err = self.block_call(cmd)
        if ret != 0:
            logger.error(f"remote sbatch failed for {task_path}: {err}")
            return None
        words = out.strip().split()
        return words[-1] if words else None

    def submit_tasks(self, task_paths: Iterable[str]) -> Dict[str, Optional[str]]:
        task_paths = [str(path) for path in task_paths]
        if not task_paths:
            return {}

        lines = [
            "set +e",
            f"cd {shlex.quote(self.remote_work_dir)} || exit 99",
        ]
        for task_path in task_paths:
            quoted_task = shlex.quote(task_path)
            lines.append(f"out=$(cd {quoted_task} && sbatch run.sbatch 2>&1)")
            lines.append("ret=$?")
            lines.append(
                "if [ $ret -eq 0 ]; then "
                f"printf 'QFLOW_SUBMIT_OK\\t%s\\t%s\\n' {shlex.quote(task_path)} \"${{out##* }}\"; "
                "else "
                f"printf 'QFLOW_SUBMIT_FAIL\\t%s\\t%s\\n' {shlex.quote(task_path)} \"$out\"; "
                "fi"
            )

        ret, out, err = self.block_call("\n".join(lines))
        if ret != 0:
            raise RuntimeError(f"remote batch sbatch failed before submission ({ret}): {err}")

        results: Dict[str, Optional[str]] = {}
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            tag, task_path, value = parts
            if tag == "QFLOW_SUBMIT_OK":
                results[task_path] = value.strip()
            elif tag == "QFLOW_SUBMIT_FAIL":
                logger.error(f"remote sbatch failed for {task_path}: {value}")
                results[task_path] = None
        for task_path in task_paths:
            results.setdefault(task_path, None)
        return results

    def active_job_ids(self, job_ids: Iterable[str]) -> Optional[set]:
        normalized = sorted({str(job_id).strip() for job_id in job_ids if str(job_id).strip()})
        if not normalized:
            return set()
        ret, out, err = self.block_call("squeue -h -o '%i' -j " + ",".join(normalized))
        if ret != 0:
            if "Invalid job id specified" in err:
                return set()
            logger.warning(f"remote squeue failed, skip running reconcile: {err}")
            return None
        return {job_id for job_id in out.split() if job_id}

    def active_job_count(self) -> Optional[int]:
        ret, out, err = self.block_call("squeue -u $USER -h -o '%i' | wc -l")
        if ret != 0:
            logger.warning(f"remote squeue count failed: {err}")
            return None
        try:
            return int(out.strip() or "0")
        except ValueError:
            logger.warning(f"remote squeue count parse failed: {out!r}")
            return None
