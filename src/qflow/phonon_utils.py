"""Phonopy工具函数模块 - 使用phonopy Python接口"""

import os
import numpy as np
from pathlib import Path
from glob import glob
from typing import Optional, Tuple, List

import yaml
from yaml import CLoader as Loader
from ase import Atoms
from ase.io import read, write
from ase.spacegroup.symmetrize import check_symmetry
from phonopy import Phonopy, PhonopyQHA, load
from phonopy.structure.atoms import PhonopyAtoms


def ase2phonopy(atoms: Atoms) -> PhonopyAtoms:
    """ASE Atoms转Phonopy PhonopyAtoms"""
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        positions=atoms.get_positions(),
    )
    return phonopy_atoms


def phonopy2ase(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """Phonopy PhonopyAtoms转ASE Atoms"""
    atoms = Atoms(
        symbols=phonopy_atoms.symbols,
        positions=phonopy_atoms.positions,
        cell=phonopy_atoms.cell,
        pbc=True
    )
    return atoms


def auto_grid_detection(atom: Atoms, max_atoms: int,
                        ratio_tolerance: float = 1.1,
                        is_sanity_check: bool = True,
                        is_verbose: bool = False) -> Tuple[int, int, int]:
    """
    自动检测超胞网格大小

    Args:
        atom: ASE atoms对象
        max_atoms: 最大原子数限制
        ratio_tolerance: 晶格比例容差
        is_sanity_check: 是否检查对称性
        is_verbose: 是否输出详细信息

    Returns:
        超胞网格 (n1, n2, n3)
    """
    lattice_vector_lengths = atom.cell.cellpar()[:3]

    # 各向同性情况
    if (lattice_vector_lengths[0] == lattice_vector_lengths[1] == lattice_vector_lengths[2]):
        number_of_replicas = int(np.round(max_atoms / len(atom)) ** (1/3))
        number_of_replicas = max(number_of_replicas, 1)
        max_replication = (number_of_replicas, number_of_replicas, number_of_replicas)
    else:
        # 各向异性情况
        lattice_vector_lengths_argsort_indices = np.argsort(lattice_vector_lengths)[::-1]
        sorted_lattice_vector_lengths = lattice_vector_lengths[lattice_vector_lengths_argsort_indices]
        ratios = [
            sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[1],
            sorted_lattice_vector_lengths[0] / sorted_lattice_vector_lengths[2],
        ]

        if ratios[0] <= ratio_tolerance and ratios[1] <= ratio_tolerance:
            number_of_replicas = int(np.round(max_atoms / len(atom)) ** (1/3))
            number_of_replicas = max(number_of_replicas, 1)
            max_replication = (number_of_replicas, number_of_replicas, number_of_replicas)
        else:
            asymmetric_replica = int((max_atoms / len(atom) / np.prod(ratios)) ** (1/3))
            asymmetric_replica = max(asymmetric_replica, 1)
            replica_r0 = max(int(np.round(asymmetric_replica * ratios[0])), 1)
            replica_r1 = max(int(np.round(asymmetric_replica * ratios[1])), 1)
            indices_to_recover = np.argsort(lattice_vector_lengths_argsort_indices)
            max_replication_arr = np.array([asymmetric_replica, replica_r0, replica_r1])[indices_to_recover]
            max_replication = tuple(max_replication_arr)

    if is_verbose:
        print(f"System: {atom}")
        print(f"Number of atoms in unit cell: {len(atom)}")
        print(f"Lattice parameters: {atom.cell.cellpar()}")

    if is_sanity_check:
        sym_unit = check_symmetry(atom, 1e-3, verbose=False)
        sym_super = check_symmetry(atom.copy().repeat(max_replication), 1e-3, verbose=False)
        # 使用属性接口避免DeprecationWarning
        symmetry_unit = getattr(sym_unit, 'international', None) or sym_unit.get("international")
        symmetry_super = getattr(sym_super, 'international', None) or sym_super.get("international")
        if symmetry_unit != symmetry_super:
            if is_verbose:
                print("Warning: Symmetry lost after replication")
            return (1, 1, 1)

    return max_replication


def get_supercell_parameters(atom: Atoms,
                             supercell_matrix: Optional[np.ndarray] = None,
                             max_atoms: Optional[int] = None,
                             min_atoms: int = 100,
                             min_length: float = 10.0) -> Tuple[int, int, int]:
    """
    获取超胞参数 - 找到满足约束的最小且均匀的超胞

    Args:
        atom: ASE atoms对象
        supercell_matrix: 用户指定的超胞矩阵
        max_atoms: 最大原子数
        min_atoms: 最小原子数（默认100）
        min_length: 最小边长（默认10 Å）

    Returns:
        超胞网格
    """
    if supercell_matrix is not None:
        return tuple(supercell_matrix)

    # 获取晶格参数
    lattice_lengths = atom.cell.cellpar()[:3]
    n_atoms = len(atom)

    # 计算每个方向需要的最小扩胞倍数（满足边长>=min_length）
    min_rep = [max(1, int(np.ceil(min_length / L))) for L in lattice_lengths]

    # 如果没有指定max_atoms，根据空间群自动设置
    if max_atoms is None:
        sym_info = check_symmetry(atom, 1e-3, verbose=False)
        space_group = getattr(sym_info, 'international', None) or sym_info.get("international")

        if space_group in ["Fd-3m", "Fm-3m", "F-43m"]:
            max_atoms = 216
        elif space_group == "P6_3mc":
            max_atoms = 450
        else:
            sorted_lengths = np.sort(lattice_lengths)[::-1]
            ratios = sorted_lengths[0] / sorted_lengths[1:]
            if np.all(ratios <= 1.1):
                max_atoms = 300
            else:
                max_atoms = 450

    # 从min_rep开始，找到满足min_atoms的最小且均匀的超胞
    # 优先选择使超胞边长最均匀的方向
    final_rep = list(min_rep)
    current_atoms = n_atoms * final_rep[0] * final_rep[1] * final_rep[2]

    while current_atoms < min_atoms:
        # 计算每个方向的当前边长
        current_lengths = [lattice_lengths[i] * final_rep[i] for i in range(3)]

        # 选择当前边长最短的方向来增加（使超胞更均匀）
        candidates = []
        for i in range(3):
            new_rep = final_rep.copy()
            new_rep[i] += 1
            new_atoms = n_atoms * new_rep[0] * new_rep[1] * new_rep[2]
            # 优先级：(当前边长, 新原子数)
            # 边长越短越优先增加，原子数相同时选择增量小的
            candidates.append((current_lengths[i], new_atoms, i))

        # 按当前边长排序（短的优先），相同长度按原子数排序
        candidates.sort(key=lambda x: (x[0], x[1]))

        # 选择第一个不超过max_atoms的候选
        best_idx = None
        for length, new_atoms, idx in candidates:
            if new_atoms <= max_atoms:
                best_idx = idx
                break

        if best_idx is None:
            # 所有方向增加都会超过max_atoms，选择当前边长最短的
            best_idx = candidates[0][2]

        final_rep[best_idx] += 1
        current_atoms = n_atoms * final_rep[0] * final_rep[1] * final_rep[2]

        # 防止无限循环
        if max(final_rep) > 20:
            break

    return tuple(final_rep)


def get_kpoints_mesh(atoms: Atoms,
                     kpoints_mesh: Optional[np.ndarray] = None,
                     kspacing: Optional[float] = None,
                     kdensity: Optional[float] = None):
    """
    获取k点网格
    优先级: kdensity > kspacing > kpoints_mesh
    """
    reciprocal_lattice = atoms.cell.reciprocal() * np.pi * 2
    reciprocal_length = [np.linalg.norm(reciprocal_lattice[:, i]) for i in range(3)]

    if not kdensity and not kspacing and kpoints_mesh is None:
        kdensity = 100.0 / 2 / np.pi

    if kdensity:
        return kdensity * 2 * np.pi
    elif kspacing:
        kpoints_mesh = []
        for i in range(3):
            kpoints_mesh.append(int(np.ceil(reciprocal_length[i] / kspacing)))
        return np.array(kpoints_mesh)
    else:
        return kpoints_mesh


def generate_phonon_displacements(atoms: Atoms,
                                  volume_dir: str,
                                  supercell: Optional[Tuple[int, int, int]] = None,
                                  max_atoms: Optional[int] = None,
                                  min_atoms: int = 100,
                                  min_length: float = 10.0,
                                  distance: float = 0.01) -> int:
    """
    生成声子位移任务

    Args:
        atoms: 优化后的ASE atoms对象
        volume_dir: 体积目录路径
        supercell: 超胞网格，如果为None则自动计算
        max_atoms: 最大原子数限制
        min_atoms: 最小原子数（默认100）
        min_length: 最小边长（默认10 Å）
        distance: 位移距离

    Returns:
        生成的任务数量
    """
    volume_dir = Path(volume_dir)
    volume_dir.mkdir(parents=True, exist_ok=True)

    if supercell is None:
        supercell = get_supercell_parameters(
            atoms,
            supercell_matrix=None,
            max_atoms=max_atoms,
            min_atoms=min_atoms,
            min_length=min_length
        )

    phonopy = Phonopy(
        ase2phonopy(atoms),
        supercell_matrix=supercell,
        primitive_matrix="auto",
        is_symmetry=True
    )

    phonopy.generate_displacements(distance=distance)

    # 生成位移任务目录
    for index, supercell_d in enumerate(phonopy.supercells_with_displacements):
        disp_atoms = Atoms(
            symbols=supercell_d.symbols,
            positions=supercell_d.positions,
            cell=supercell_d.cell,
            pbc=True
        )
        task_dir = volume_dir / f"task.{index:06d}"
        task_dir.mkdir(exist_ok=True)
        write(str(task_dir / "POSCAR"), disp_atoms, format="vasp", vasp5=True)

    # 保存未位移的完美超胞
    perfect_supercell = phonopy.supercell
    perfect_atoms = Atoms(
        symbols=perfect_supercell.symbols,
        positions=perfect_supercell.positions,
        cell=perfect_supercell.cell,
        pbc=True
    )
    task_perfect_dir = volume_dir / "task_perfect"
    task_perfect_dir.mkdir(exist_ok=True)
    write(str(task_perfect_dir / "POSCAR"), perfect_atoms, format="vasp", vasp5=True)

    # 保存phonopy位移信息
    analyze_dir = volume_dir / "analyze"
    analyze_dir.mkdir(exist_ok=True)
    phonopy.save(str(analyze_dir / "phonopy_disp.yaml"))

    return len(phonopy.supercells_with_displacements)


def collect_forces(volume_dir: str, use_vasprun: bool = False) -> List[np.ndarray]:
    """
    收集力数据

    Args:
        volume_dir: 体积目录路径
        use_vasprun: 是否从vasprun.xml读取（默认False，使用forces.txt）

    Returns:
        力数组列表
    """
    task_dirs = sorted(glob(f"{volume_dir}/task.[0-9]*"))
    forces = []

    for task_dir in task_dirs:
        # 优先使用forces.txt（MatterSim输出）
        forces_file = os.path.join(task_dir, "forces.txt")
        if os.path.exists(forces_file):
            forces.append(np.loadtxt(forces_file))
        elif use_vasprun:
            # 回退到vasprun.xml
            from pymatgen.io.vasp import Vasprun
            vasprun_file = os.path.join(task_dir, "vasprun.xml")
            if os.path.exists(vasprun_file):
                vasprun = Vasprun(vasprun_file, parse_dos=False, parse_eigen=False)
                forces.append(vasprun.ionic_steps[-1]['forces'])

    return forces


def postprocess_phonon(volume_dir: str,
                       t_min: int = 0,
                       t_max: int = 2000,
                       t_step: int = 10,
                       use_vasprun: bool = False) -> bool:
    """
    声子后处理

    Args:
        volume_dir: 体积目录路径
        t_min: 最低温度
        t_max: 最高温度
        t_step: 温度步长
        use_vasprun: 是否从vasprun.xml读取力（默认False，使用forces.txt）

    Returns:
        True表示有虚频，False表示无虚频
    """
    volume_dir = Path(volume_dir)
    analyze_dir = volume_dir / "analyze"

    # 读取原胞结构
    struct_dir = volume_dir.parent
    poscar_file = struct_dir / "POSCAR"
    if not poscar_file.exists():
        # 尝试从opt目录读取
        poscar_file = struct_dir / "opt" / "CONTCAR"
    atoms = read(str(poscar_file))

    # 加载phonopy
    phonopy = load(str(analyze_dir / "phonopy_disp.yaml"))

    # 收集力
    forces = collect_forces(str(volume_dir), use_vasprun=use_vasprun)
    phonopy.forces = forces

    # 计算力常数
    phonopy.produce_force_constants()
    phonopy.symmetrize_force_constants()

    # 计算k点网格
    kpoints_mesh = get_kpoints_mesh(atoms, kpoints_mesh=None, kspacing=None, kdensity=None)

    # 运行mesh计算
    phonopy.run_mesh(kpoints_mesh, is_mesh_symmetry=True)

    # 绘制能带和DOS
    try:
        band_fig = phonopy.auto_band_structure(plot=True, write_yaml=True)
        band_fig.savefig(str(analyze_dir / "phonon_band.png"), dpi=300)
        band_fig.clf()
    except Exception as e:
        print(f"Warning: Failed to plot band structure: {e}")

    try:
        dos_fig = phonopy.auto_total_dos(plot=True, write_dat=True)
        dos_fig.savefig(str(analyze_dir / "phonon_dos.png"), dpi=300)
        dos_fig.clf()
    except Exception as e:
        print(f"Warning: Failed to plot DOS: {e}")

    # 计算热力学性质
    phonopy.run_mesh(kpoints_mesh)
    phonopy.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)

    # 保存结果
    phonopy.write_yaml_thermal_properties(filename=str(analyze_dir / "thermo_properties.yaml"))
    phonopy.save(str(analyze_dir / "phonopy_params.yaml"))

    # 检查虚频
    band_dict = phonopy.get_band_structure_dict()
    frequencies = band_dict['frequencies']
    has_imaginary = any((np.array(block) < -1e-3).any() for block in frequencies)

    return has_imaginary


def check_imaginary_frequency(volume_dir: str) -> bool:
    """
    检查是否存在虚频

    Args:
        volume_dir: 体积目录路径

    Returns:
        True表示有虚频，False表示无虚频
    """
    analyze_dir = Path(volume_dir) / "analyze"
    phonopy_params = analyze_dir / "phonopy_params.yaml"

    if not phonopy_params.exists():
        # 没有后处理结果，无法判断
        return False

    try:
        phonopy = load(str(phonopy_params))
        band_dict = phonopy.get_band_structure_dict()
        if band_dict is None:
            # 需要重新计算
            kpoints_mesh = 50  # 使用默认值
            phonopy.run_mesh(kpoints_mesh)
            phonopy.auto_band_structure()
            band_dict = phonopy.get_band_structure_dict()

        frequencies = band_dict['frequencies']
        has_imaginary = any((np.array(block) < -1e-3).any() for block in frequencies)
        return has_imaginary
    except Exception as e:
        print(f"Warning: Failed to check imaginary frequency: {e}")
        return False


def collect_energies_natoms(struct_dir: str,
                            volumes: List[float],
                            use_vasprun: bool = False) -> Tuple[np.ndarray, int]:
    """
    收集各体积点的能量和原子数

    Args:
        struct_dir: 结构目录
        volumes: 体积列表
        use_vasprun: 是否从vasprun.xml读取（默认False，使用energy.txt）

    Returns:
        (能量数组, 原子数)
    """
    energies = []
    natoms = None

    for volume in volumes:
        volume_dir = Path(struct_dir) / f"volume_{volume}"

        # 优先使用energy.txt（MatterSim输出）
        energy_file = volume_dir / "task_perfect" / "energy.txt"
        if not energy_file.exists():
            energy_file = volume_dir / "analyze" / "energy.txt"

        if energy_file.exists():
            energies.append(float(energy_file.read_text().strip()))
        elif use_vasprun:
            from pymatgen.io.vasp import Vasprun
            vasprun_file = volume_dir / "task_perfect" / "vasprun.xml"
            if vasprun_file.exists():
                vasprun = Vasprun(str(vasprun_file), parse_dos=False, parse_eigen=False)
                energies.append(vasprun.final_energy)

        # 获取原子数
        if natoms is None:
            poscar = volume_dir / "task_perfect" / "POSCAR"
            if poscar.exists():
                atoms = read(str(poscar))
                natoms = len(atoms)

    return np.array(energies), natoms


def postprocess_qha(struct_dir: str,
                    volumes: List[float],
                    pressure: float = 0,
                    t_min: int = 0,
                    t_max: int = 1000,
                    t_step: int = 10,
                    use_vasprun: bool = False):
    """
    QHA后处理

    Args:
        struct_dir: 结构目录
        volumes: 体积列表
        pressure: 压力 (GPa)
        t_min: 最低温度
        t_max: 最高温度
        t_step: 温度步长
        use_vasprun: 是否从vasprun.xml读取（默认False）
    """
    struct_dir = Path(struct_dir)
    temperatures = np.arange(t_min, t_max + 1, t_step)

    # 收集能量
    displaced_energies, natoms = collect_energies_natoms(
        str(struct_dir), volumes, use_vasprun=use_vasprun
    )

    # 收集热力学性质
    cv_list, entropy_list, free_energy_list = [], [], []
    real_volumes = []

    for volume in volumes:
        volume_dir = struct_dir / f"volume_{volume}"
        yaml_file = volume_dir / "analyze" / "thermo_properties.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(f"找不到 {yaml_file}")

        with open(yaml_file) as f:
            thermal_properties = yaml.load(f, Loader=Loader)["thermal_properties"]

        cv_list.append([v["heat_capacity"] for v in thermal_properties])
        entropy_list.append([v["entropy"] for v in thermal_properties])
        free_energy_list.append([v["free_energy"] for v in thermal_properties])

        # 获取真实体积
        poscar = volume_dir / "task_perfect" / "POSCAR"
        atoms = read(str(poscar))
        real_volumes.append(atoms.get_volume())

    real_volumes = np.array(real_volumes)
    cv = np.array(cv_list)
    entropy = np.array(entropy_list)
    free_energy = np.array(free_energy_list)

    # 创建QHA对象
    qha = PhonopyQHA(
        free_energy=free_energy.T,
        volumes=real_volumes,
        temperatures=temperatures,
        t_max=t_max,
        cv=cv.T,
        entropy=entropy.T,
        electronic_energies=displaced_energies,
        pressure=pressure,
    )

    # 保存结果
    analyze_dir = struct_dir / "analyze"
    analyze_dir.mkdir(exist_ok=True)

    prefix = f"pressure_{pressure:.2f}_"

    # 保存数据文件
    qha.write_gibbs_temperature(filename=str(analyze_dir / f"{prefix}gibbs-temperature.dat"))
    qha.write_volume_temperature(filename=str(analyze_dir / f"{prefix}volume-temperature.dat"))
    qha.write_bulk_modulus_temperature(filename=str(analyze_dir / f"{prefix}bulk_modulus-temperature.dat"))
    qha.write_heat_capacity_P_numerical(filename=str(analyze_dir / f"{prefix}Cp-temperature.dat"))
    qha.write_gruneisen_temperature(filename=str(analyze_dir / f"{prefix}gruneisen-temperature.dat"))
    qha.write_helmholtz_volume(filename=str(analyze_dir / f"{prefix}helmholtz-volume.dat"))

    # 保存每原子Gibbs能
    gibbs_energies = qha.get_gibbs_temperature()
    gibbs_per_atom = gibbs_energies / natoms
    mask = temperatures <= t_max
    actual_temperatures = temperatures[mask][:len(gibbs_per_atom)]

    gibbs_file = analyze_dir / f"{prefix}gibbs_per_atom-temperature.dat"
    gibbs_data = np.column_stack([actual_temperatures, gibbs_per_atom[:len(actual_temperatures)]])
    np.savetxt(str(gibbs_file), gibbs_data,
               header='Temperature(K)  Gibbs_Energy_per_atom(eV/atom)',
               fmt='%.6f')

    # 保存图表
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = qha.plot_bulk_modulus_temperature()
        if fig:
            fig.savefig(str(analyze_dir / f"{prefix}bulk_modulus-temperature.png"), dpi=300)
            plt.close(fig)

        fig = qha.plot_volume_temperature()
        if fig:
            fig.savefig(str(analyze_dir / f"{prefix}volume-temperature.png"), dpi=300)
            plt.close(fig)

        fig = qha.plot_thermal_expansion()
        if fig:
            fig.savefig(str(analyze_dir / f"{prefix}thermal_expansion.png"), dpi=300)
            plt.close(fig)

        fig = qha.plot_gibbs_temperature()
        if fig:
            fig.savefig(str(analyze_dir / f"{prefix}gibbs-temperature.png"), dpi=300)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to plot QHA figures: {e}")

    print(f"QHA后处理完成: {struct_dir}")


# ============== BTE (phono3py) 相关函数 ==============

def generate_bte_displacements(atoms: Atoms,
                               bte_dir: str,
                               supercell: Optional[Tuple[int, int, int]] = None,
                               max_atoms: Optional[int] = None,
                               min_atoms: int = 100,
                               min_length: float = 10.0,
                               distance: float = 0.03,
                               is_plusminus: bool = True,
                               symprec: float = 1e-3) -> dict:
    """
    生成 BTE (phono3py) fc2 + fc3 位移任务

    目录结构:
        bte_dir/
        ├── fc2/
        │   ├── task.000000/POSCAR   # fc2 位移
        │   ├── ...
        │   └── task_perfect/POSCAR  # 无位移超胞
        ├── fc3/
        │   ├── task.000000/POSCAR   # fc3 位移
        │   ├── ...
        │   └── task_perfect/POSCAR
        └── analyze/
            └── phono3py_disp.yaml   # phono3py 位移信息

    Args:
        atoms: 优化后的 ASE atoms 对象
        bte_dir: BTE 工作目录
        supercell: 超胞网格，None 则自动计算
        max_atoms: 最大原子数
        min_atoms: 最小原子数
        min_length: 最小边长 (Å)
        distance: 位移距离 (Å)
        is_plusminus: 是否使用正负位移
        symprec: 对称性精度

    Returns:
        {'n_fc2': int, 'n_fc3': int, 'supercell': tuple, 'n_atoms_sc': int}
    """
    from phono3py import Phono3py

    bte_dir = Path(bte_dir)
    bte_dir.mkdir(parents=True, exist_ok=True)

    if supercell is None:
        supercell = get_supercell_parameters(
            atoms,
            supercell_matrix=None,
            max_atoms=max_atoms,
            min_atoms=min_atoms,
            min_length=min_length
        )

    sc_matrix = np.diag(supercell) if isinstance(supercell, tuple) else supercell

    phonopy_atoms = ase2phonopy(atoms)

    ph3 = Phono3py(
        phonopy_atoms,
        supercell_matrix=sc_matrix,
        primitive_matrix='auto',
        phonon_supercell_matrix=sc_matrix,
        is_symmetry=True,
        symprec=symprec,
        is_mesh_symmetry=False,
        log_level=0,
    )
    ph3.generate_displacements(distance=distance, is_plusminus=is_plusminus)

    fc2_supercells = ph3.phonon_supercells_with_displacements
    fc3_supercells = ph3.supercells_with_displacements
    n_fc2 = len(fc2_supercells)
    n_fc3 = len(fc3_supercells)

    # ---- fc2 位移 ----
    fc2_dir = bte_dir / 'fc2'
    fc2_dir.mkdir(exist_ok=True)

    # 无位移参考超胞
    perfect_sc2 = ph3.phonon_supercell
    _write_phonopy_poscar(perfect_sc2, fc2_dir / 'task_perfect')

    for i, sc in enumerate(fc2_supercells):
        task_dir = fc2_dir / f'task.{i:06d}'
        task_dir.mkdir(exist_ok=True)
        _write_phonopy_poscar(sc, task_dir)

    # ---- fc3 位移 ----
    fc3_dir = bte_dir / 'fc3'
    fc3_dir.mkdir(exist_ok=True)

    # 无位移参考超胞
    perfect_sc3 = ph3.supercell
    _write_phonopy_poscar(perfect_sc3, fc3_dir / 'task_perfect')

    for i, sc in enumerate(fc3_supercells):
        task_dir = fc3_dir / f'task.{i:06d}'
        task_dir.mkdir(exist_ok=True)
        _write_phonopy_poscar(sc, task_dir)

    # ---- 保存 phono3py 位移信息 ----
    analyze_dir = bte_dir / 'analyze'
    analyze_dir.mkdir(exist_ok=True)
    ph3.save(str(analyze_dir / 'phono3py_disp.yaml'),
             settings={"force_sets": False, "displacements": True})

    return {
        'n_fc2': n_fc2,
        'n_fc3': n_fc3,
        'supercell': tuple(np.diag(sc_matrix).astype(int)),
        'n_atoms_sc': len(ph3.supercell),
    }


def _write_phonopy_poscar(phonopy_atoms, task_dir: Path):
    """将 PhonopyAtoms 写为 POSCAR 到 task_dir/POSCAR"""
    task_dir = Path(task_dir)
    task_dir.mkdir(exist_ok=True)
    atoms = phonopy2ase(phonopy_atoms)
    write(str(task_dir / 'POSCAR'), atoms, format='vasp', vasp5=True)


def collect_bte_forces(bte_dir: str, fc_type: str = 'fc2',
                       use_vasprun: bool = True) -> List[np.ndarray]:
    """
    收集 BTE 位移的力数据

    Args:
        bte_dir: BTE 工作目录
        fc_type: 'fc2' 或 'fc3'
        use_vasprun: 是否从 vasprun.xml 读取

    Returns:
        力数组列表
    """
    fc_dir = Path(bte_dir) / fc_type
    task_dirs = sorted(fc_dir.glob('task.[0-9]*'))
    forces = []

    for task_dir in task_dirs:
        forces_file = task_dir / 'forces.txt'
        if forces_file.exists():
            forces.append(np.loadtxt(str(forces_file)))
        elif use_vasprun:
            vasprun_file = task_dir / 'vasprun.xml'
            if vasprun_file.exists():
                from pymatgen.io.vasp import Vasprun
                vasprun = Vasprun(str(vasprun_file), parse_dos=False, parse_eigen=False)
                forces.append(np.array(vasprun.ionic_steps[-1]['forces']))
            else:
                raise FileNotFoundError(f"No forces found in {task_dir}")
        else:
            raise FileNotFoundError(f"No forces found in {task_dir}")

    return forces


def postprocess_bte(bte_dir: str,
                    tmin: float = 50,
                    tmax: float = 500,
                    tstep: float = 50,
                    method: str = 'RTA',
                    is_isotope: bool = True,
                    target_length: float = 60.0,
                    max_mesh: int = 24,
                    use_vasprun: bool = True) -> dict:
    """
    BTE 后处理：收集力 → 计算力常数 → 计算热导率

    Args:
        bte_dir: BTE 工作目录
        tmin, tmax, tstep: 温度范围 (K)
        method: 'RTA' 或 'LBTE'
        is_isotope: 是否包含同位素散射
        target_length: q-mesh 实空间采样长度 (Å)
        max_mesh: q-mesh 每方向最大值
        use_vasprun: 是否从 vasprun.xml 读取力

    Returns:
        {'kappa_300K': float, 'temperatures': list, 'kappa_values': list,
         'has_imaginary': bool, 'min_freq': float}
    """
    from phono3py import Phono3py
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    from phonopy import Phonopy

    bte_dir = Path(bte_dir)
    analyze_dir = bte_dir / 'analyze'

    # 1. 加载 phono3py 位移信息
    disp_yaml = analyze_dir / 'phono3py_disp.yaml'
    if not disp_yaml.exists():
        raise FileNotFoundError(f"phono3py_disp.yaml not found in {analyze_dir}")

    from phono3py import load as phono3py_load
    ph3 = phono3py_load(str(disp_yaml), log_level=0)

    # 2. 收集 fc2 力
    print("Collecting fc2 forces...")
    fc2_forces = collect_bte_forces(str(bte_dir), 'fc2', use_vasprun=use_vasprun)
    ph3.phonon_forces = fc2_forces
    ph3.produce_fc2(symmetrize_fc2=True)

    # 3. 检查虚频
    print("Checking imaginary frequencies...")
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

    # 保存虚频检查结果
    import json
    imag_result = {
        'has_imaginary': has_imaginary,
        'min_freq_THz': min_freq,
        'n_imaginary': int(np.sum(freqs_all < -0.1)),
    }
    with open(analyze_dir / 'imaginary_check.json', 'w') as f:
        json.dump(imag_result, f, indent=2)

    if has_imaginary:
        print(f"WARNING: Structure has imaginary frequencies (min={min_freq:.4f} THz)")
        # 保存 fc2 并返回
        write_fc2_to_hdf5(
            ph3.fc2,
            p2s_map=ph3.phonon_primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            filename=str(analyze_dir / 'fc2.hdf5'),
        )
        return {
            'kappa_300K': None,
            'temperatures': [],
            'kappa_values': [],
            'has_imaginary': True,
            'min_freq': min_freq,
        }

    # 4. 收集 fc3 力
    print("Collecting fc3 forces...")
    fc3_forces = collect_bte_forces(str(bte_dir), 'fc3', use_vasprun=use_vasprun)
    ph3.forces = fc3_forces
    ph3.produce_fc3(symmetrize_fc3r=True)

    # 保存力常数
    write_fc2_to_hdf5(
        ph3.fc2,
        p2s_map=ph3.phonon_primitive.p2s_map,
        physical_unit="eV/angstrom^2",
        filename=str(analyze_dir / 'fc2.hdf5'),
    )
    write_fc3_to_hdf5(
        ph3.fc3,
        p2s_map=ph3.phonon_primitive.p2s_map,
        filename=str(analyze_dir / 'fc3.hdf5'),
    )

    # 5. 计算 q-mesh
    prim_atoms = phonopy2ase(ph3.primitive)
    lengths = prim_atoms.cell.cellpar()[:3]
    mesh = []
    for L in lengths:
        N = int(np.ceil(target_length / L))
        N = max(1, min(N, max_mesh))
        mesh.append(N)
    # 对称性修正
    tol = 0.05
    if abs(lengths[0] - lengths[1]) / max(lengths[0], lengths[1]) < tol:
        mesh[0] = mesh[1] = max(mesh[0], mesh[1])
    if (abs(lengths[0] - lengths[1]) / max(lengths[0], lengths[1]) < tol and
        abs(lengths[1] - lengths[2]) / max(lengths[1], lengths[2]) < tol):
        mesh[:] = [max(mesh)] * 3

    print(f"Q-mesh: {mesh}")

    # 6. 计算热导率
    temperatures = np.arange(tmin, tmax + tstep, tstep)
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(symmetrize_fc3q=False)

    if method.upper() == 'RTA':
        ph3.run_thermal_conductivity(
            temperatures=temperatures,
            is_isotope=is_isotope,
            write_kappa=True,
            conductivity_type="wigner"
        )
    else:
        ph3.run_thermal_conductivity(
            temperatures=temperatures,
            is_LBTE=True,
            is_isotope=is_isotope,
            write_kappa=True,
            conductivity_type="wigner",
            boundary_mfp=1e6
        )

    # 移动 kappa HDF5 到 analyze 目录
    for kf in Path('.').glob('kappa-*.hdf5'):
        kf.rename(analyze_dir / kf.name)

    # 7. 提取 kappa 数据
    tc = ph3.thermal_conductivity
    kappa_tensor = tc.kappa  # shape: (n_temps, 6) 或 (n_temps, 3, 3)
    temps = tc.temperatures

    # 提取 xx, yy, zz 分量并取平均
    if kappa_tensor.ndim == 3:
        kappa_iso = (kappa_tensor[:, 0, 0] + kappa_tensor[:, 1, 1] + kappa_tensor[:, 2, 2]) / 3.0
    else:
        kappa_iso = (kappa_tensor[:, 0] + kappa_tensor[:, 1] + kappa_tensor[:, 2]) / 3.0

    # 找 300K 的 kappa
    idx_300 = np.argmin(np.abs(temps - 300))
    kappa_300 = float(kappa_iso[idx_300])

    # 保存 kappa 汇总
    kappa_file = analyze_dir / 'kappa_summarize.dat'
    with open(kappa_file, 'w') as f:
        f.write("# T(K)  kappa_xx  kappa_yy  kappa_zz  kappa_iso\n")
        for i, T in enumerate(temps):
            if kappa_tensor.ndim == 3:
                kxx, kyy, kzz = kappa_tensor[i, 0, 0], kappa_tensor[i, 1, 1], kappa_tensor[i, 2, 2]
            else:
                kxx, kyy, kzz = kappa_tensor[i, 0], kappa_tensor[i, 1], kappa_tensor[i, 2]
            f.write(f"{T:7.1f}  {kxx:10.3f}  {kyy:10.3f}  {kzz:10.3f}  {kappa_iso[i]:10.3f}\n")

    # 保存 phono3py 完整结果
    ph3.save(str(analyze_dir / 'phono3py_params.yaml'),
             settings={"force_sets": True, "force_constants": True})

    print(f"BTE post-processing done. kappa(300K) = {kappa_300:.2f} W/mK")

    return {
        'kappa_300K': kappa_300,
        'temperatures': temps.tolist(),
        'kappa_values': kappa_iso.tolist(),
        'has_imaginary': False,
        'min_freq': min_freq,
    }
