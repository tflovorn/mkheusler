import argparse
import os
import numpy as np
from ase.lattice import bulk
from mkheusler.pwscf.build import build_pw2wan, build_bands, build_qe
from mkheusler.wannier.build import Winfile
from mkheusler.build.util import _base_dir, _global_config

def verify_SC10_fcc(system, a):
    SC10_fcc = np.array([[0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]])

    if not np.array_equal(system.cell, SC10_fcc):
        raise ValueError("system does not follow Setyawan FCC cell convention")

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

def make_qe_config(system, latconst, soc, num_bands, ecutwfc, ecutrho, degauss, Nk, band_path):
    base = _base_dir()
    if soc:
        pseudo_dir = os.path.join(base, "pseudo", "SG15", "soc")
    else:
        pseudo_dir = os.path.join(base, "pseudo", "SG15", "base_no_soc")

    pseudo = get_pseudo(system.get_chemical_symbols())
    weight = get_weight(system)
    conv_thr = {"scf": 1e-8, "nscf": 1e-10, "bands": 1e-10}

    qe_config = {"pseudo_dir": pseudo_dir, "pseudo": pseudo, "soc": soc, "latconst": latconst, 
            "num_bands": num_bands, "weight": weight, "ecutwfc": ecutwfc, "ecutrho": ecutrho,
            "degauss": degauss, "conv_thr": conv_thr, "Nk": Nk, "band_path": band_path}

    return qe_config

def write_qe_input(prefix, file_dir, qe_input, calc_type):
    file_path = os.path.join(file_dir, "{}.{}.in".format(prefix, calc_type))
    with open(file_path, 'w') as fp:
        fp.write(qe_input[calc_type])

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
    args = parser.parse_args()

    # TODO intial magnetic moment specifiers?

    # NOTE - assuming no SOC = SG15 and SOC = SG15 plus SOC.
    if args.ecutwfc is None:
        ecutwfc = 60.0
    else:
        ecutwfc = args.ecutwfc

    if args.ecutrho is None:
        ecutrho = 240.0
    else:
        ecutrho = args.ecutrho

    atoms = args.atoms.split(',')
    if len(atoms) == 3:
        system_type = "HH"
        wan_valence = {atoms[0]: "spd", atoms[1]: "spd", atoms[2]: "sp"}
        if args.soc:
            num_wann = 44
        else:
            num_wann = 22

        if args.prefix is None:
            prefix = "{}{}{}_bulk".format(atoms[0], atoms[1], atoms[2])
            if args.soc:
                prefix = "{}_soc".format(prefix)
    elif len(atoms) == 4:
        system_type = "FH"
        wan_valence = {atoms[0]: "spd", atoms[2]: "spd", atoms[3]: "sp"}
        if args.soc:
            num_wann = 62
        else:
            num_wann = 31

        if args.prefix is None:
            prefix = "{}2{}{}_bulk".format(atoms[0], atoms[2], atoms[3])
            if args.soc:
                prefix = "{}_soc".format(prefix)
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

    # TODO may want to revise num_wann and num_bands values
    num_bands = 2*num_wann

    band_path_syms = ["Gamma", "X", "W", "K", "Gamma", "L", "U", "W", "L", "K", "W", "U", "X"]
    SC10_kpts = {"Gamma": np.array([0.0, 0.0, 0.0]),
            "K": np.array([3/8, 3/8, 3/4]),
            "L": np.array([1/2, 1/2, 1/2]),
            "U": np.array([5/8, 1/4, 5/8]),
            "W": np.array([1/2, 1/4, 3/4]),
            "X": np.array([1/2, 0.0, 1/2])}
    band_path = [SC10_kpts[sym] for sym in band_path_syms]

    Nk = {"scf": args.Nk_scf, "nscf": args.Nk_nscf, "bands": args.Nk_bands}
    qe_config = make_qe_config(system, args.latconst, args.soc, num_bands, ecutwfc,
            ecutrho, args.degauss, Nk, band_path)

    qe_input = {}
    for calc_type in ["scf", "nscf", "bands"]:
        qe_input[calc_type] = build_qe(system, prefix, calc_type, qe_config)

    work = os.path.join(_global_config()["work_base"], prefix)
    if not os.path.exists(work):
        os.mkdir(work)

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

    wannier_input = Winfile(system, qe_config, wan_valence, num_wann)
    win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
    with open(win_path, 'w') as fp:
        fp.write(wannier_input)

if __name__ == "__main__":
    _main()
