from __future__ import division
import argparse
import os
from copy import deepcopy
import numpy as np
from ase.lattice import bulk
from mkheusler.pwscf.build import build_pw2wan, build_bands, build_qe
from mkheusler.wannier.build import Winfile
from mkheusler.queue.queuefile import write_queuefile
from mkheusler.build.util import _base_dir, _global_config

def verify_SC10_fcc(system, a):
    SC10_fcc = np.array([[0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]])

    if not np.array_equal(system.cell, SC10_fcc):
        raise ValueError("system does not follow Setyawan FCC cell convention")

def sc10_fcc_path_syms():
    # SC10 path
    band_path_syms = ["Gamma", "X", "W", "K", "Gamma", "L", "U", "W", "L", "K", "W", "U", "X"]
    band_path_labels = ["$\\Gamma$", "$X$", "$W$", "$K$", "$\\Gamma$", "$L$", "$U$",
            "$W$", "$L$", "$K$", "$W$", "$U$", "$X$"]

    return band_path_syms, band_path_labels

def get_num_bands(system, system_type, atoms, soc):
    # TODO may want to revise output from this fn based on projections
    syms = system.get_chemical_symbols()
    if system_type == "HH":
        wann_valence = {atoms[0]: 9, atoms[1]: 9, atoms[2]: 4}
    elif system_type == "FH":
        # for FH, atoms[0] = atoms[1] (TODO - check here?)
        wann_valence = {atoms[0]: 9, atoms[2]: 9, atoms[3]: 4}
    else:
        raise ValueError("found unexpected system_type")

    num_wann = 0
    for sym in syms:
        if sym in wann_valence:
            num_wann += wann_valence[sym]
        else:
            raise ValueError("found unexpected atomic symbol")

    if soc:
        num_wann *= 2

    extra_bands_factor = 2.5
    num_bands = int(np.ceil(extra_bands_factor*num_wann))
    if num_bands % 2 == 1:
        num_bands += 1

    return num_wann, num_bands

def get_weight(system):
    syms = system.get_chemical_symbols()
    all_weights = system.get_masses()

    weight = {}
    for sym, sym_weight in zip(syms, all_weights):
        weight[sym] = sym_weight

    return weight

def get_pseudo(syms):
    pseudo = {}
    # TODO want to search pseudo_dir instead?
    for sym in syms:
        pseudo[sym] = "{}.UPF".format(sym)

    return pseudo

def get_cutoff(ecutwfc_in, ecutrho_in):
    if ecutwfc_in is None:
        ecutwfc = 60.0
    else:
        ecutwfc = args.ecutwfc

    if ecutrho_in is None:
        ecutrho = 240.0
    else:
        ecutrho = args.ecutrho

    return ecutwfc, ecutrho

def get_pseudo_dir(soc, sg15_adjust):
    base = _base_dir()
    if soc:
        pseudo_dir = os.path.join(base, "pseudo", "SG15", "adjusted_soc")
    elif sg15_adjust:
        pseudo_dir = os.path.join(base, "pseudo", "SG15", "adjusted_no_soc")
    else:
        pseudo_dir = os.path.join(base, "pseudo", "SG15", "base_no_soc")

    return pseudo_dir

def make_qe_config(system, latconst, soc, magnetic, num_bands, ecutwfc, ecutrho, degauss, Nk, band_path, pseudo_dir):
    pseudo = get_pseudo(system.get_chemical_symbols())
    weight = get_weight(system)
    conv_thr = {"scf": 1e-8, "nscf": 1e-10, "bands": 1e-10}

    qe_config = {"pseudo_dir": pseudo_dir, "pseudo": pseudo, "soc": soc, "magnetic": magnetic, "latconst": latconst, 
            "num_bands": num_bands, "weight": weight, "ecutwfc": ecutwfc, "ecutrho": ecutrho,
            "degauss": degauss, "conv_thr": conv_thr, "Nk": Nk, "band_path": band_path}

    return qe_config

def write_qe_input(prefix, file_dir, qe_input, calc_type):
    file_path = os.path.join(file_dir, "{}.{}.in".format(prefix, calc_type))
    with open(file_path, 'w') as fp:
        fp.write(qe_input[calc_type])

def get_work(prefix=None):
    gconf = _global_config()
    work = os.path.expandvars(gconf["work_base"])
    if prefix is not None:
        work = os.path.join(work, prefix)

    if not os.path.exists(work):
        os.mkdir(work)

    return work

def _write_queuefiles(work, prefix, config, mpi_tasks_per_node):
    wan_setup_config = deepcopy(config)
    wan_setup_config["calc"] = "wan_setup"
    write_queuefile(wan_setup_config)

    pw_post_config = deepcopy(config)
    pw_post_config["calc"] = "pw_post"
    pw_post_config["nodes"] = 1
    pw_post_config["mpi_tasks"] = mpi_tasks_per_node // config["openmp_threads_per_mpi_task"]
    write_queuefile(pw_post_config)

    wan_run_config = deepcopy(pw_post_config)
    wan_run_config["calc"] = "wan_run"
    write_queuefile(wan_run_config)

def _machine_settings(machine):
    if machine == "stampede2":
        num_nodes = 1
        mpi_tasks_per_node = 68
        total_mpi_tasks = num_nodes * mpi_tasks_per_node
        openmp_threads_per_mpi_task = 1

        # QE pools = number of k-points run in parallel.
        # Must have total_mpi_tasks divisible by total_pools.
        pools_per_node = 17
        total_pools = num_nodes * pools_per_node
    elif machine == "ls5":
        num_nodes = 1
        mpi_tasks_per_node = 24
        total_mpi_tasks = num_nodes * mpi_tasks_per_node
        openmp_threads_per_mpi_task = 1

        # QE pools = number of k-points run in parallel.
        # Must have total_mpi_tasks divisible by total_pools.
        pools_per_node = 4
        total_pools = num_nodes * pools_per_node

    return num_nodes, mpi_tasks_per_node, total_mpi_tasks, openmp_threads_per_mpi_task, total_pools

def _main():
    parser = argparse.ArgumentParser("Build and run Heusler bulk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("atoms", type=str,
            help="Comma-separated list of XYZ/X2YZ atoms (ex: 'Ni,Mn,Sb' or 'Co,Co,Mn,Si')")
    parser.add_argument("latconst", type=float,
            help="Lattice constant in Ang (conventional cell cubic edge)")
    parser.add_argument("--prefix", type=str, default=None,
            help="System name (obtained from atoms if not specified)")
    parser.add_argument("--soc", action="store_true",
            help="Use spin-orbit coupling")
    parser.add_argument("--magnetic", action="store_true",
            help="Start in magnetized state")
    parser.add_argument("--ecutwfc", type=float, default=None,
            help="Wavefunction plane-wave cutoff energy (Ry)")
    parser.add_argument("--ecutrho", type=float, default=None,
            help="Charge density plane-wave cutoff energy (Ry)")
    parser.add_argument("--degauss", type=float, default=0.02,
            help="Delta-function smearing constant (Ry)")
    parser.add_argument("--Nk_scf", type=int, default=16,
            help="Number of k-points to use in SCF calculation (same in each direction)")
    parser.add_argument("--Nk_nscf", type=int, default=8,
            help="Number of k-points to use in NSCF calculation (same in each direction)")
    parser.add_argument("--Nk_bands", type=int, default=20,
            help="Number of k-points to use for each panel in bands calculation")
    parser.add_argument("--sg15_adjust", action="store_true",
            help="If specified for non-SOC calculation, use SG15 PPs adjusted to work for SOC")
    args = parser.parse_args()

    # TODO intial magnetic moment specifiers?

    # NOTE - assuming no SOC = SG15 and SOC = SG15 plus SOC.
    ecutwfc, ecutrho = get_cutoff(args.ecutwfc, args.ecutrho)

    atoms = args.atoms.split(',')
    if len(atoms) == 3:
        system_type = "HH"
        wann_valence = {atoms[0]: "spd", atoms[1]: "spd", atoms[2]: "sp"}

        if args.prefix is None:
            prefix = "{}{}{}_bulk".format(atoms[0], atoms[1], atoms[2])
            if args.soc:
                prefix = "{}_soc".format(prefix)
            elif args.sg15_adjust:
                prefix = "{}_adjust".format(prefix)
        else:
            prefix = args.prefix
    elif len(atoms) == 4:
        system_type = "FH"
        wann_valence = {atoms[0]: "spd", atoms[2]: "spd", atoms[3]: "sp"}

        if args.prefix is None:
            prefix = "{}2{}{}_bulk".format(atoms[0], atoms[2], atoms[3])
            if args.soc:
                prefix = "{}_soc".format(prefix)
            elif args.sg15_adjust:
                prefix = "{}_adjust".format(prefix)
        else:
            prefix = args.prefix
    else:
        raise ValueError("must specify 3 or 4 atoms (half-Heusler or full-Heusler)")

    system = bulk(atoms, 'fcc', a=args.latconst)
    verify_SC10_fcc(system, args.latconst)

    # SC10 = Setyawan and Curtarolo, Comp. Mater. Sci. 49, 299 (2010).
    # SC10 FCC cell 111 = cubic conventional cell 111
    # --> Scaled positions along body diagonal are same as in
    # cubic conventional cell.
    if system_type == "HH":
        # Half-Heusler: Y-X-Z-void
        system.set_scaled_positions([[1/4, 1/4, 1/4],
                [0.0, 0.0, 0.0],
                [1/2, 1/2, 1/2]])
    elif system_type == "FH":
        # Full-Heusler: Y-X-Z-X
        system.set_scaled_positions([[1/4, 1/4, 1/4],
            [3/4, 3/4, 3/4],
            [0.0, 0.0, 0.0],
            [1/2, 1/2, 1/2]])

    num_wann, num_bands = get_num_bands(system, system_type, atoms, args.soc)

    band_path_syms, band_path_labels = sc10_fcc_path_syms()
    SC10_kpts = {"Gamma": np.array([0.0, 0.0, 0.0]),
            "K": np.array([3/8, 3/8, 3/4]),
            "L": np.array([1/2, 1/2, 1/2]),
            "U": np.array([5/8, 1/4, 5/8]),
            "W": np.array([1/2, 1/4, 3/4]),
            "X": np.array([1/2, 0.0, 1/2])}
    band_path = [SC10_kpts[sym] for sym in band_path_syms]

    pseudo_dir = get_pseudo_dir(args.soc, args.sg15_adjust)

    Nk = {"scf": [args.Nk_scf]*3, "nscf": [args.Nk_nscf]*3, "bands": args.Nk_bands}
    qe_config = make_qe_config(system, args.latconst, args.soc, args.magnetic,
            num_bands, ecutwfc, ecutrho, args.degauss, Nk, band_path, pseudo_dir)

    qe_input = {}
    for calc_type in ["scf", "nscf", "bands"]:
        qe_input[calc_type] = build_qe(system, prefix, calc_type, qe_config)

    work = get_work(prefix)
    wannier_dir = os.path.join(work, "wannier")
    if not os.path.exists(wannier_dir):
        os.mkdir(wannier_dir)

    bands_dir = os.path.join(work, "bands")
    if not os.path.exists(bands_dir):
        os.mkdir(bands_dir)

    write_qe_input(prefix, wannier_dir, qe_input, "scf")
    write_qe_input(prefix, wannier_dir, qe_input, "nscf")
    write_qe_input(prefix, bands_dir, qe_input, "bands")

    pw2wan_input = build_pw2wan(prefix, args.soc)
    pw2wan_path = os.path.join(wannier_dir, "{}.pw2wan.in".format(prefix))
    with open(pw2wan_path, 'w') as fp:
        fp.write(pw2wan_input)

    bands_post_input = build_bands(prefix)
    bands_post_path = os.path.join(bands_dir, "{}.bands_post.in".format(prefix))
    with open(bands_post_path, 'w') as fp:
        fp.write(bands_post_input)

    wannier_input = Winfile(system, qe_config, wann_valence, num_wann)
    win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
    with open(win_path, 'w') as fp:
        fp.write(wannier_input)

    machine = "ls5"
    (num_nodes, mpi_tasks_per_node, total_mpi_tasks, openmp_threads_per_mpi_task,
            total_pools) = _machine_settings(machine)

    queue_config = {"machine": machine, "queue": "normal", "max_jobs": 1,
            "nodes": num_nodes, "mpi_tasks": total_mpi_tasks,
            "openmp_threads_per_mpi_task": openmp_threads_per_mpi_task,
            "qe_pools": total_pools,
            "hours": 4, "minutes": 0, "wannier": True, "project": "A-ph911",
            "prefix": prefix, "base_path": get_work(),
            "outer_min": -12.0, "outer_max": 16.0,
            "inner_min": -12.0, "inner_max": 14.0}

    _write_queuefiles(work, prefix, queue_config, mpi_tasks_per_node)

if __name__ == "__main__":
    _main()
