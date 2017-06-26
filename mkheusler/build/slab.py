from __future__ import division
import argparse
import os
import numpy as np
from ase import Atoms
from mkheusler.pwscf.build import build_pw2wan, build_bands, build_qe
from mkheusler.wannier.build import Winfile
from mkheusler.build.util import _base_dir, _global_config
from mkheusler.build.bulk import (verify_SC10_fcc, get_num_bands, get_cutoff, get_pseudo_dir,
        make_qe_config, write_qe_input, get_work, _write_queuefiles, _machine_settings)

def slab_fcc_111_path_syms():
    # Assumes SC10 fcc lattice and ASE [111] slab which has
    # D = sqrt(2)a/2 [1 1/2
    #                 0 sqrt(3)/2]
    band_path_syms = ["Gamma", "M", "K", "Gamma"]
    band_path_labels = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]

    return band_path_syms, band_path_labels

def get_system_type(atoms):
    if len(atoms) == 3:
        system_type = "HH"
    elif len(atoms) == 4:
        system_type = "FH"
    else:
        raise ValueError("must specify 3 or 4 atoms (half-Heusler or full-Heusler)")

    return system_type

def get_band_path(surface_normal_cubic):
    # TODO consider surfaces other than 111
    if surface_normal_cubic == (1, 1, 1):
        band_path_syms, band_path_labels = slab_fcc_111_path_syms()
        fcc_111_kpts = {"Gamma": np.array([0.0, 0.0, 0.0]),
                "K": np.array([2/3, 2/3, 0.0]),
                "M": np.array([1/2, 0.0, 0.0])}
        band_path = [fcc_111_kpts[sym] for sym in band_path_syms]
    else:
        raise ValueError("unsupported surface direction (need band path)")

    return band_path, band_path_labels

def _get_cutoffs(surface_termination, layers):
    if surface_termination == '1':
        top_cutoff = 0.0
        bottom_cutoff = (4 * (layers - 1) + 2) / 4
        new_layers = layers
    elif surface_termination == '2':
        top_cutoff = 1/4
        bottom_cutoff = (4 * (layers - 1) + 2 + 1/3) / 4
        new_layers = layers + 1
    elif surface_termination == '3':
        top_cutoff = (2/3) / 4
        bottom_cutoff = (4 * (layers - 1) + 1 + 1/3) / 4
        new_layers = layers
    else:
        raise ValueError("unrecognized surface termination")

    return top_cutoff, bottom_cutoff, new_layers

def _outside_cutoff(top_cutoff, bottom_cutoff, a3_val):
    eps = 1e-9
    above = top_cutoff - a3_val > eps
    below = a3_val - bottom_cutoff > eps

    return above or below

def make_surface_system(atoms, latconst, layers, surface_normal_cubic, vacuum, surface_termination):
    '''Create an ASE atoms object representing a half-Heusler or full-Heusler slab.

    `atoms` is a length-3 (half-Heusler) or 4 (full-Heusler) list of atomic symbols in XXYZ order
    (note: atoms along the body diagonal appear in YXZX order).
    `latconst` is the cubic lattice constant in Angstrom.
    `layers` is the number of 3- or 4-atom layer groups for atoms in the A positions,
    before `top_cut` and `bottom_cut` are applied.
    `surface_normal_cubic` specifies the normal vector to the surface.
        TODO: use this. For now, the value given is ignored and [111] is assumed.
    `vacuum`: amount of vacuum to apply between slabs.
    `surface_termination`: for [111], valid values are: '1', '2', '3'.
        These specify the size of the nearest-neighbor atom groups at the surfaces
        in the half-Heusler structure.
        Here A, B, C are the three configurations of the triangular lattice generated
        by moving along the body diagonal in the fcc structure.
        '1': A Y ~ C Z - A X - B Y (top, moving down) / A Z ~ B Y - A X - C Z (bottom, moving up).
        '2': A X - B Y ~ A Z - B X / B X - A Z ~ B Y - A X
        '3': C Z - A X - B Y ~ A Z / B Y - A X - C Z ~ A Y
    '''
    system_type = get_system_type(atoms)

    a1 = (latconst/2) * np.array([1.0, -1.0, 0.0])
    a2 = (latconst/2) * np.array([-1.0, 0.0, 1.0])
    a3_base = latconst * np.array([1.0, 1.0, 1.0])

    planar_pos = {'A': np.array([0.0, 0.0, 0.0]),
            'B': (2/3)*a1 + (1/3)*a2,
            'C': (1/3)*a1 + (2/3)*a2}
    a3_ABC_pos = {'A': 0.0, 'B': 1/3, 'C': 2/3}

    if system_type == "HH":
        # Half-Heusler: Y-X-Z-void
        a3_displacements = [1/4, 0.0, 1/2]
    elif system_type == "FH":
        # Full-Heusler: Y-X-Z-X
        a3_displacements = [1/4, 3/4, 0.0, 1/2]
    else:
        raise ValueError("unsupported system_type")

    top_cutoff, bottom_cutoff, layers = _get_cutoffs(surface_termination, layers)

    slab_atoms = []
    slab_cart_pos = []
    a3_vals = []
    for layer_index in range(layers):
        for at, at_a3_val in zip(atoms, a3_displacements):
            for tri_base in ['A', 'B', 'C']:
                a3_val_in_layer = (at_a3_val + a3_ABC_pos[tri_base]) % 1
                a3_val = a3_val_in_layer + layer_index

                if _outside_cutoff(top_cutoff, bottom_cutoff, a3_val):
                    continue

                slab_atoms.append(at)
                a3_vals.append(a3_val)
                slab_cart_pos.append(planar_pos[tri_base] + a3_val * a3_base)

    top_a3 = max(a3_vals)
    a3_length = top_a3 * np.linalg.norm(a3_base) + vacuum
    a3 = (a3_length / np.linalg.norm(a3_base)) * a3_base

    system_slab = Atoms(slab_atoms, slab_cart_pos, cell=[a1, a2, a3],
            pbc=[True, True, True])

    return system_slab

def make_prefix(atoms, layers, soc, surface_termination):
    # TODO include growth dir in prefix?
    if len(atoms) == 3:
        prefix = "{}{}{}_slab_{}_surf_{}".format(atoms[0], atoms[1], atoms[2], layers, surface_termination)
        if soc:
            prefix = "{}_soc".format(prefix)
    elif len(atoms) == 4:
        prefix = "{}2{}{}_slab_{}_surf_{}".format(atoms[0], atoms[2], atoms[3], layers, surface_termination)
        if soc:
            prefix = "{}_soc".format(prefix)
    else:
        raise ValueError("must specify 3 or 4 atoms (half-Heusler or full-Heusler)")

    return prefix

def _main():
    parser = argparse.ArgumentParser("Build and run Heusler slab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("atoms", type=str,
            help="Comma-separated list of XYZ/X2YZ atoms (ex: 'Ni,Mn,Sb' or 'Co,Co,Mn,Si')")
    parser.add_argument("latconst", type=float,
            help="Lattice constant in Ang (conventional cell cubic edge)")
    parser.add_argument("layers", type=int,
            help="Number of layers in the slab")
    parser.add_argument("--surface_termination", type=str, default='1',
            help="Type of surface termination (described in make_surface_system)")
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
            help="Number of k-points to use in SCF calculation (same in each direction along slab)")
    parser.add_argument("--Nk_nscf", type=int, default=8,
            help="Number of k-points to use in NSCF calculation (same in each direction along slab)")
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
        wann_valence = {atoms[0]: "spd", atoms[1]: "spd", atoms[2]: "sp"}
    elif len(atoms) == 4:
        wann_valence = {atoms[0]: "spd", atoms[2]: "spd", atoms[3]: "sp"}
    else:
        raise ValueError("must specify 3 or 4 atoms (half-Heusler or full-Heusler)")

    if args.prefix is None:
        prefix = make_prefix(atoms, args.layers, args.soc, args.surface_termination)
        if args.sg15_adjust:
            prefix = "{}_adjust".format(prefix)
    else:
        prefix = args.prefix

    vacuum = 20 # Angstrom
    surface_normal_cubic = (1, 1, 1)

    system_slab = make_surface_system(atoms, args.latconst, args.layers, surface_normal_cubic, vacuum, args.surface_termination)

    system_type = get_system_type(atoms)
    num_wann, num_bands = get_num_bands(system_slab, system_type, atoms, args.soc)

    band_path, band_path_labels = get_band_path(surface_normal_cubic)

    pseudo_dir = get_pseudo_dir(args.soc, args.sg15_adjust)

    Nk = {"scf": [args.Nk_scf, args.Nk_scf, 1],
            "nscf": [args.Nk_nscf, args.Nk_nscf, 1],
            "bands": args.Nk_bands}
    qe_config = make_qe_config(system_slab, args.latconst, args.soc, args.magnetic, num_bands, ecutwfc,
            ecutrho, args.degauss, Nk, band_path, pseudo_dir)

    qe_input = {}
    for calc_type in ["scf", "nscf", "bands"]:
        qe_input[calc_type] = build_qe(system_slab, prefix, calc_type, qe_config)

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

    wannier_input = Winfile(system_slab, qe_config, wann_valence, num_wann)
    win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
    with open(win_path, 'w') as fp:
        fp.write(wannier_input)

    machine = "ls5"
    num_nodes = 6
    (mpi_tasks_per_node, total_mpi_tasks, openmp_threads_per_mpi_task,
            total_pools) = _machine_settings(machine, num_nodes)

    queue_config = {"machine": machine, "queue": "normal", "max_jobs": 1,
            "nodes": num_nodes, "mpi_tasks": total_mpi_tasks,
            "openmp_threads_per_mpi_task": openmp_threads_per_mpi_task,
            "qe_pools": total_pools,
            "hours": 24, "minutes": 0, "wannier": False, "project": "A-ph911",
            "prefix": prefix, "base_path": get_work(),
            "outer_min": -12.0, "outer_max": 16.0,
            "inner_min": -12.0, "inner_max": 14.0}

    _write_queuefiles(work, prefix, queue_config, mpi_tasks_per_node)

if __name__ == "__main__":
    _main()
