from __future__ import division, print_function
import argparse
import json
import numpy as np
from ase import Atoms
from ase.dft.kpoints import bandpath
from ase.parallel import parprint, paropen
import ase.io
from gpaw import GPAW, MethfesselPaxton
from gpaw.utilities import h2gpts
from gpaw_run_util.filenames import standard_filenames
from gpaw_run_util.lcao_bands import run_bands
from mkheusler.build.slab import make_prefix, make_surface_system

def make_parallel_scf():
    return {'kpt':         None,
            'domain':              None,
            'band':                1, # want to increase this? default is just one. how to pick good value?
            'order':               'kdb',
            'stridebands':         False,
            'sl_auto':             True,
            'sl_default':          None,
            'sl_diagonalize':      None,
            'sl_inverse_cholesky': None,
            'sl_lcao':             None,
            'sl_lrtddft':          None,
            'buffer_size':         None}

def _main():
    parser = argparse.ArgumentParser("Build and run Heusler slab",
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
    args = parser.parse_args()

    atoms = args.atoms.split(',')

    prefix = make_prefix(atoms, args.layers, args.soc)
    filenames = standard_filenames(prefix)
    vacuum = 20 # Angstrom
    surface_normal_cubic = (1, 1, 1)

    slab = make_surface_system(atoms, args.latconst, args.layers, surface_normal_cubic, vacuum)
    slab.set_pbc((True, True, False))

    parallel_scf = make_parallel_scf()

    ase.io.write(filenames['structure_xsf'], slab, format='xsf')

    # Perform self-consistent calculation.
    h = 0.2 # FD grid spacing (Angstrom). Complains about Poisson grid if too small (h = 0.1).
    smearing_width = 0.1 # eV
    Nk_scf = 12

    # LCAO documentation strongly encorages ensuring grid points are divisible by 8
    # to improve performance.
    gpts = h2gpts(h, slab.get_cell(), idiv=8)

    calc = GPAW(mode='lcao',
            basis='dzp',
            gpts=gpts,
            xc='PBE',
            kpts=(Nk_scf, Nk_scf, 1),
            occupations=MethfesselPaxton(smearing_width),
            random=True,
            parallel=parallel_scf,
            txt=filenames['scf_calc_out'])

    slab.set_calculator(calc)
    E_gs = slab.get_potential_energy()
    calc.write(filenames['scf_restart'])
    E_F = calc.get_fermi_level()

    parprint("E_gs = {}".format(E_gs))
    parprint("E_F = {}".format(E_F))

    # Perform bands calculation.
    band_path, band_path_labels = get_band_path(surface_normal_cubic)
    ks_per_panel = 60
    band_ks, band_xs, band_special_xs = bandpath(band_path, slab.get_cell(), len(path)*ks_per_panel + 1)

    run_bands(prefix, E_F, band_ks, band_xs, band_special_xs, band_path_labels)

if __name__ == "__main__":
    _main()
