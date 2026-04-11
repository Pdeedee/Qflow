"""轻量级提交任务扫描与注册。"""

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class SubmitTaskScanner:
    """快速扫描可提交目录并提取任务元数据。"""

    def __init__(self, work_dir: Path, structures_dir: Path):
        self.work_dir = Path(work_dir).resolve()
        self.structures_dir = Path(structures_dir).resolve()

        structures_relpath = self.structures_dir.relative_to(self.work_dir).as_posix()
        base = rf"^{re.escape(structures_relpath)}/(?P<structure_name>[^/]+)"

        self._bte_task_pattern = re.compile(
            rf"{base}/(?P<pressure_name>P_\d+GPa)/bte/(?P<fc>fc2|fc3)/(?P<task_name>task\.[^/]+|task_perfect)$"
        )
        self._bte_opt_pattern = re.compile(
            rf"{base}/(?P<pressure_name>P_\d+GPa)/opt$"
        )
        self._volume_task_pattern = re.compile(
            rf"{base}/(?P<volume_name>volume_[^/]+)/(?P<task_name>task\.[^/]+|task_perfect)$"
        )
        self._volume_opt_pattern = re.compile(
            rf"{base}/(?P<volume_name>volume_[^/]+)/opt$"
        )
        self._opt_pattern = re.compile(
            rf"{base}(?:/[^/]+)*/opt$"
        )
        self._plain_task_pattern = re.compile(
            rf"{base}(?:/[^/]+)*/(?P<task_name>task\.[^/]+|task_perfect)$"
        )

    def is_submit_candidate_name(self, name: str, plain_only: bool) -> bool:
        if plain_only:
            return name.startswith('task.')
        return name == 'opt' or name == 'task_perfect' or name.startswith('task.')

    def _classify_submit_candidate(self, task_path: str) -> Optional[Dict]:
        normalized_path = task_path.replace(os.sep, '/')

        match = self._bte_task_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': f"bte_{data['fc']}",
                'structure_name': data['structure_name'],
                'volume_name': None,
                'pressure_name': data['pressure_name'],
            }

        match = self._bte_opt_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': 'bte_opt',
                'structure_name': data['structure_name'],
                'volume_name': None,
                'pressure_name': data['pressure_name'],
            }

        match = self._volume_task_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': 'phonon' if data['volume_name'] == 'volume_1.0' else 'qha',
                'structure_name': data['structure_name'],
                'volume_name': data['volume_name'],
                'pressure_name': None,
            }

        match = self._volume_opt_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': 'opt' if data['volume_name'] == 'volume_1.0' else 'qha_opt',
                'structure_name': data['structure_name'],
                'volume_name': data['volume_name'],
                'pressure_name': None,
            }

        match = self._opt_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': 'opt',
                'structure_name': data['structure_name'],
                'volume_name': None,
                'pressure_name': None,
            }

        match = self._plain_task_pattern.match(normalized_path)
        if match:
            data = match.groupdict()
            return {
                'path': task_path,
                'task_type': 'plain',
                'structure_name': data['structure_name'],
                'volume_name': None,
                'pressure_name': None,
            }

        return None

    def iter_scan(self, plain_only: bool = False) -> Iterable[Dict]:
        """使用 os.scandir 递归扫描候选任务目录。"""
        if not self.structures_dir.exists():
            return

        scan_stack = [self.structures_dir]
        while scan_stack:
            current_dir = scan_stack.pop()
            for entry in os.scandir(current_dir):
                if not entry.is_dir(follow_symlinks=False):
                    continue

                if self.is_submit_candidate_name(entry.name, plain_only):
                    rel_path = Path(entry.path).relative_to(self.work_dir).as_posix()
                    record = self._classify_submit_candidate(rel_path)
                    if record is not None:
                        yield record
                    continue

                scan_stack.append(Path(entry.path))

    def scan(self, plain_only: bool = False) -> List[Dict]:
        """返回扫描结果列表。"""
        return list(self.iter_scan(plain_only=plain_only))
