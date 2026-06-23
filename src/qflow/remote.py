# -*- coding: utf-8 -*-
"""Remote SLURM submission support for QFlow."""

import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import paramiko

from .logger import logger


def normalize_job_id(job_id: str) -> str:
    """Normalize Slurm job ids for DB/squeue comparisons."""
    job_id = str(job_id or "").strip()
    if not job_id:
        return ""
    return job_id.split(".", 1)[0].split("_", 1)[0]


def _run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as proc:
        out, err = proc.communicate()
        return proc.returncode, out, err


def rsync(
    from_path: str,
    to_path: str,
    port: int = 22,
    key_filename: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 20,
    option_args: Optional[List[str]] = None,
    additional_args: Optional[List[str]] = None,
):
    ssh_cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        "-q",
    ]
    if password:
        ssh_cmd.extend(["-o", "BatchMode=no"])
    else:
        ssh_cmd.extend(["-o", "BatchMode=yes"])
    if key_filename:
        ssh_cmd.extend(["-i", key_filename])

    cmd = ["rsync", "-az", "-e", " ".join(ssh_cmd), "-q"]
    env = None
    if password:
        if shutil.which("sshpass") is None:
            raise RuntimeError("rsync password login requires sshpass in PATH")
        cmd = ["sshpass", "-e"] + cmd
        env = os.environ.copy()
        env["SSHPASS"] = password
    if option_args:
        cmd.extend(option_args)
    cmd.extend([from_path, to_path])
    if additional_args:
        cmd.extend(additional_args)

    ret, out, err = _run_cmd(cmd, env=env)
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
        self.rclone_remote = str(self.remote_config.get("rclone_remote") or "").rstrip(":")
        self.key_filename = self.remote_config.get("ssh_key_filename")
        self.password = self._resolve_password()
        self._rsync_key_filename = self.key_filename
        self._use_sftp_transfer = False
        self.ssh = None
        self._sftp = None
        self.remotename = f"{self.username}@{self.hostname}"

        if not self.hostname or not self.username or not self.remote_root:
            raise ValueError("remote mode requires remote.ssh_hostname, remote.ssh_username, and remote.remote_root")

        self._setup_ssh()

    def _resolve_password(self) -> Optional[str]:
        password = self.remote_config.get("ssh_password")
        if password:
            return str(password)

        password_env = self.remote_config.get("ssh_password_env")
        if password_env:
            return os.environ.get(str(password_env))

        return None

    def _setup_ssh(self):
        key_candidates = [] if self.password and not self.key_filename else self._ssh_key_candidates()
        if self.password:
            key_candidates.append(None)
        if not key_candidates:
            key_candidates = [None]

        errors = []
        for key_filename in key_candidates:
            use_password = key_filename is None and self.password is not None
            key_msg = "password" if use_password else (key_filename or "default SSH keys")
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
                        password=self.password if use_password else None,
                        timeout=self.timeout,
                        compress=True,
                        allow_agent=False,
                        look_for_keys=(key_filename is None and not use_password),
                    )
                except Exception as exc:
                    errors.append(f"{key_msg}: {type(exc).__name__}: {exc!r}")
                    logger.warning(f"remote ssh key failed: {key_msg}: {type(exc).__name__}: {exc!r}")
                    self.ssh.close()
                    time.sleep(2)
                    continue
                self.key_filename = key_filename
                self._rsync_key_filename = None if use_password else key_filename
                self._use_sftp_transfer = use_password and shutil.which("sshpass") is None
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

    def _rclone_target(self, path: str = "") -> str:
        path = str(path).strip("/")
        suffix = f"/{path}" if path else ""
        return f"{self.rclone_remote}:{self.remote_work_dir}{suffix}"

    def _use_rclone(self) -> bool:
        return bool(self.rclone_remote and shutil.which("rclone"))

    def _rclone_copy_files(self, files: Iterable[str]):
        files = [str(path).rstrip("/") for path in files]
        if not files:
            return
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            files_from = fp.name
            for path in files:
                fp.write(path + "\n")
        try:
            cmd = [
                "rclone", "copy",
                str(self.work_dir),
                self._rclone_target(),
                "--files-from", files_from,
                "--create-empty-src-dirs",
                "--transfers", str(self.remote_config.get("rclone_transfers", 16)),
                "--checkers", str(self.remote_config.get("rclone_checkers", 32)),
                "--fast-list",
                "--copy-links",
            ]
            ret, _out, err = _run_cmd(cmd)
            if ret != 0:
                raise RuntimeError(f"rclone upload failed: {' '.join(cmd)}\n{err.decode('utf-8', errors='replace')}")
        finally:
            try:
                os.remove(files_from)
            except OSError:
                pass

    def _rclone_fetch_dirs(self, task_paths: Iterable[str]):
        for task_path in task_paths:
            task_path = str(task_path).strip("/")
            if not task_path:
                continue
            local_path = self.work_dir / task_path
            local_path.mkdir(parents=True, exist_ok=True)
            cmd = [
                "rclone", "copy",
                self._rclone_target(task_path),
                str(local_path),
                "--exclude", "POTCAR",
                "--exclude", "*.xml",
                "--transfers", str(self.remote_config.get("rclone_transfers", 16)),
                "--checkers", str(self.remote_config.get("rclone_checkers", 32)),
                "--fast-list",
            ]
            ret, _out, err = _run_cmd(cmd)
            if ret != 0:
                raise RuntimeError(f"rclone fetch failed: {' '.join(cmd)}\n{err.decode('utf-8', errors='replace')}")

    def _get_sftp(self):
        self.ensure_alive()
        if self._sftp is None:
            self._sftp = self.ssh.open_sftp()
        return self._sftp

    def _sftp_mkdir_p(self, remote_dir: str):
        sftp = self._get_sftp()
        parts = []
        current = str(remote_dir).strip("/")
        while current:
            parts.append(current)
            current = str(Path(current).parent)
            if current == ".":
                break
        for part in reversed(parts):
            path = "/" + part
            try:
                sftp.stat(path)
            except FileNotFoundError:
                sftp.mkdir(path)

    def _sftp_put_files(self, files: Iterable[str]):
        sftp = self._get_sftp()
        self._sftp_mkdir_p(self.remote_work_dir)
        for rel_path in files:
            local_path = self.work_dir / rel_path
            if not local_path.is_file():
                continue
            remote_path = f"{self.remote_work_dir}/{rel_path}"
            self._sftp_mkdir_p(str(Path(remote_path).parent))
            sftp.put(str(local_path), remote_path)

    def _sftp_fetch_dir(self, remote_dir: str, local_dir: Path):
        sftp = self._get_sftp()
        local_dir.mkdir(parents=True, exist_ok=True)
        for item in sftp.listdir_attr(remote_dir):
            name = item.filename
            if name == "POTCAR" or name.endswith(".xml"):
                continue
            remote_path = f"{remote_dir.rstrip('/')}/{name}"
            local_path = local_dir / name
            if item.st_mode & 0o040000:
                self._sftp_fetch_dir(remote_path, local_path)
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path, str(local_path))

    def rsync_send_files(self, files: Iterable[str]):
        files = self._expand_files_from(files)
        if not files:
            return
        self.block_checkcall(f"mkdir -p {shlex.quote(self.remote_work_dir)}")
        if self._use_rclone():
            self._rclone_copy_files(files)
            return
        if self._use_sftp_transfer:
            self._sftp_put_files(files)
            return
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            files_from = fp.name
            for path in files:
                fp.write(path.rstrip("/") + "\n")
        try:
            rsync(
                str(self.work_dir) + "/",
                f"{self.remotename}:{self.remote_work_dir}/",
                port=self.port,
                key_filename=self._rsync_key_filename,
                password=self.password,
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
        if self._use_rclone():
            self._rclone_fetch_dirs([task_path])
            return
        if self._use_sftp_transfer:
            self._sftp_fetch_dir(remote_path.rstrip("/"), local_path)
            return
        rsync(
            f"{self.remotename}:{remote_path}",
            str(local_path) + "/",
            port=self.port,
            key_filename=self._rsync_key_filename,
            password=self.password,
            timeout=self.timeout,
            additional_args=["--exclude=POTCAR", "--exclude=*.xml"],
        )

    def rsync_fetch_tasks(self, task_paths: Iterable[str]):
        task_paths = [str(path).rstrip("/") for path in task_paths]
        if not task_paths:
            return
        if self._use_rclone():
            self._rclone_fetch_dirs(task_paths)
            return
        if self._use_sftp_transfer:
            for task_path in task_paths:
                self.rsync_fetch_task(task_path)
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
                key_filename=self._rsync_key_filename,
                password=self.password,
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
            lines.append(f"out=$((cd {quoted_task} && sbatch run.sbatch) 2>&1)")
            lines.append("ret=$?")
            lines.append(
                "if [ $ret -eq 0 ]; then "
                f"printf 'QFLOW_SUBMIT_OK\\t%s\\t%s\\n' {shlex.quote(task_path)} \"${{out##* }}\"; "
                "else "
                "[ -n \"$out\" ] || out=\"exit_code=$ret with no output\"; "
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
                error_path = self.work_dir / task_path / "remote_submit_error.log"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(value + "\n")
                results[task_path] = None
        for task_path in task_paths:
            results.setdefault(task_path, None)
        return results

    def active_job_ids(self, job_ids: Iterable[str]) -> Optional[set]:
        normalized = sorted({normalize_job_id(job_id) for job_id in job_ids if normalize_job_id(job_id)})
        if not normalized:
            return set()
        ret, out, err = self.block_call("squeue -h -o '%i' -j " + ",".join(normalized))
        if ret != 0:
            if "Invalid job id specified" in err:
                active_ids = set()
                for job_id in normalized:
                    single_ret, single_out, single_err = self.block_call(
                        "squeue -h -o '%i' -j " + shlex.quote(job_id)
                    )
                    if single_ret == 0:
                        active_ids.update(
                            normalize_job_id(item)
                            for item in single_out.split()
                            if normalize_job_id(item)
                        )
                    elif "Invalid job id specified" not in single_err:
                        logger.warning(f"remote squeue failed for job {job_id}: {single_err}")
                        return None
                return active_ids
            logger.warning(f"remote squeue failed, skip running reconcile: {err}")
            return None
        return {normalize_job_id(job_id) for job_id in out.split() if normalize_job_id(job_id)}

    def cancel_jobs(self, job_ids: Iterable[str]) -> int:
        normalized = sorted({normalize_job_id(job_id) for job_id in job_ids if normalize_job_id(job_id)})
        if not normalized:
            return 0
        ret, _out, err = self.block_call("scancel " + " ".join(shlex.quote(job_id) for job_id in normalized))
        if ret != 0:
            raise RuntimeError(f"remote scancel failed: {err}")
        return len(normalized)

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
