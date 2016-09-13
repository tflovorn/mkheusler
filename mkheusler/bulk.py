import argparse
import numpy as np
from ase.lattice import bulk

def verify_SC10_fcc(system, a):
    SC10_fcc = np.array([[0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]])

    if not np.array_equal(system.cell, SC10_fcc):
        raise ValueError("system does not follow Setyawan FCC cell convention")

def _main():
    parser = argparse.ArgumentParser("Build and run Heusler bulk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("atoms", type=str,
            help="Comma-separated list of XYZ/X2YZ atoms (ex: 'Ni,Mn,Sb' or 'Co,Co,Mn,Si')")
    parser.add_argument("latconst", type=float,
            help="Lattice constant (conventional cell cubic edge)")
    parser.add_argument("--system_name", type=str, default=None,
            help="System name (obtained from atoms if not specified)")
    parser.add_argument("--soc", action="store_true",
            help="Use spin-orbit coupling")
    parser.add_argument("--ecutwfc", type=float, default=None,
            help="Wavefunction plane-wave cutoff energy (Ry)")
    parser.add_argument("--ecutrho", type=float, default=None,
            help="Charge density plane-wave cutoff energy (Ry)")
    parser.add_argument("--sigma", type=float, default=0.02,
            help="Delta-function smearing constant (Ry)")
    parser.add_argument("--Nk_scf", type=int, default=16,
            help="Number of k-points to use in SCF calculation (same in each direction)")
    parser.add_argument("--Nk_nscf", type=int, default=8,
            help="Number of k-points to use in NSCF calculation (same in each direction)")
    parser.add_argument("--Nk_bands", type=int, default=20,
            help="Number of k-points to use for each panel in bands calculation")
    args = parser.parse_args()

    # TODO intial magnetic moment specifiers?
    # TODO handle cutoff defaults / SOC / PP interaction

    atoms = args.atoms.split(',')
    if len(atoms) == 3:
        system_type = "HH"
        if args.system_name is None:
            system_name = "{}{}{}_bulk".format(atoms[0], atoms[1], atoms[2])
    elif len(atoms) == 4:
        system_type = "FH"
        if args.system_name is None:
            system_name = "{}2{}{}_bulk".format(atoms[0], atoms[2], atoms[3])
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

    print(system.get_chemical_symbols())
    print(system.get_masses())
    print(system.get_scaled_positions())
    print(system.get_initial_magnetic_moments())
    print(system.get_tags())

    kpath_syms = ["Gamma", "X", "W", "K", "Gamma", "L", "U", "W", "L", "K", "W", "U", "X"]
    SC10_kpts = {"Gamma": np.array([0.0, 0.0, 0.0]),
            "K": np.array([3/8, 3/8, 3/4]),
            "L": np.array([1/2, 1/2, 1/2]),
            "U": np.array([5/8, 1/4, 5/8]),
            "W": np.array([1/2, 1/4, 3/4]),
            "X": np.array([1/2, 0.0, 1/2])}
    kpath = [SC10_kpts[sym] for sym in kpath_syms]

if __name__ == "__main__":
    _main()
