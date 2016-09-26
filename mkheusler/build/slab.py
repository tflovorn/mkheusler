import argparse
import os
import numpy as np
from ase.lattice import bulk
from ase.build import surface
from mkheusler.pwscf.build import build_pw2wan, build_bands, build_qe
from mkheusler.wannier.build import Winfile
from mkheusler.build.util import _base_dir, _global_config
from mkheusler.build.bulk import verify_SC10_fcc, get_num_bands, get_cutoff, make_qe_config, write_qe_input

def _main():
    parser = argparse.ArgumentParser("Build and run Heusler bulk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("atoms", type=str,
            help="Comma-separated list of XYZ/X2YZ atoms (ex: 'Ni,Mn,Sb' or 'Co,Co,Mn,Si')")
    parser.add_argument("latconst", type=float,
            help="Lattice constant in Ang (conventional cell cubic edge)")
    parser.add_argument("layers", type=int,
            help="Number of layers in the slab")
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
            help="Number of k-points to use in SCF calculation (same in each direction along slab)")
    parser.add_argument("--Nk_nscf", type=int, default=8,
            help="Number of k-points to use in NSCF calculation (same in each direction along slab)")
    parser.add_argument("--Nk_bands", type=int, default=20,
            help="Number of k-points to use for each panel in bands calculation")
    args = parser.parse_args()

    # TODO intial magnetic moment specifiers?

    # NOTE - assuming no SOC = SG15 and SOC = SG15 plus SOC.
    ecutwfc, ecutrho = get_cutoff(args.ecutwfc, args.ecutrho)

    # TODO include growth dir in prefix?
    atoms = args.atoms.split(',')
    if len(atoms) == 3:
        system_type = "HH"
        wann_valence = {atoms[0]: "spd", atoms[1]: "spd", atoms[2]: "sp"}

        if args.prefix is None:
            prefix = "{}{}{}_slab_{}".format(atoms[0], atoms[1], atoms[2], args.layers)
            if args.soc:
                prefix = "{}_soc".format(prefix)
    elif len(atoms) == 4:
        system_type = "FH"
        wann_valence = {atoms[0]: "spd", atoms[2]: "spd", atoms[3]: "sp"}

        if args.prefix is None:
            prefix = "{}2{}{}_slab_{}".format(atoms[0], atoms[2], atoms[3], args.layers)
            if args.soc:
                prefix = "{}_soc".format(prefix)
    else:
        raise ValueError("must specify 3 or 4 atoms (half-Heusler or full-Heusler)")

    system_bulk = bulk(atoms, 'fcc', a=args.latconst)
    verify_SC10_fcc(system_bulk, args.latconst)

    # SC10 = Setyawan and Curtarolo, Comp. Mater. Sci. 49, 299 (2010).
    # SC10 FCC cell 111 = cubic conventional cell 111
    # --> Scaled positions along body diagonal are same as in
    # cubic conventional cell.
    if system_type == "HH":
        # Half-Heusler: Y-X-Z-void
        system_bulk.set_scaled_positions([[1/4, 1/4, 1/4],
                [0.0, 0.0, 0.0],
                [1/2, 1/2, 1/2]])
    elif system_type == "FH":
        # Full-Heusler: Y-X-Z-X
        system_bulk.set_scaled_positions([[1/4, 1/4, 1/4],
            [3/4, 3/4, 3/4],
            [0.0, 0.0, 0.0],
            [1/2, 1/2, 1/2]])

    surface_normal_cubic = (1, 1, 1)
    surface_normal_fcc = (1, 1, 1) # TODO convert from cubic to fcc system (111 is the same)
    vacuum = 20 # Angstrom
    system_slab = surface(system_bulk, surface_normal_fcc, args.layers, vacuum)

    num_wann, num_bands = get_num_bands(system_slab, system_type, atoms, args.soc)

    # TODO (111) surface
    band_path_syms = ["Gamma", "X", "W", "K", "Gamma", "L", "U", "W", "L", "K", "W", "U", "X"]
    SC10_kpts = {"Gamma": np.array([0.0, 0.0, 0.0]),
            "K": np.array([3/8, 3/8, 3/4]),
            "L": np.array([1/2, 1/2, 1/2]),
            "U": np.array([5/8, 1/4, 5/8]),
            "W": np.array([1/2, 1/4, 3/4]),
            "X": np.array([1/2, 0.0, 1/2])}
    band_path = [SC10_kpts[sym] for sym in band_path_syms]

    Nk = {"scf": args.Nk_scf, "nscf": args.Nk_nscf, "bands": args.Nk_bands}
    qe_config = make_qe_config(system_slab, args.latconst, args.soc, num_bands, ecutwfc,
            ecutrho, args.degauss, Nk, band_path)

    qe_input = {}
    for calc_type in ["scf", "nscf", "bands"]:
        qe_input[calc_type] = build_qe(system_slab, prefix, calc_type, qe_config)

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

    wannier_input = Winfile(system_slab, qe_config, wann_valence, num_wann)
    win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
    with open(win_path, 'w') as fp:
        fp.write(wannier_input)

if __name__ == "__main__":
    _main()