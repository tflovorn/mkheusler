import argparse
import os
from mkheusler.build.bulk import get_work, sc10_fcc_path_syms
from mkheusler.pwscf.extractQEBands import extractQEBands
from mkheusler.pwscf.parseScf import fermi_from_scf, alat_from_scf, latVecs_from_scf
from mkheusler.wannier.extractHr import extractHr
from mkheusler.wannier.plotBands import plotBands

def _main():
    parser = argparse.ArgumentParser("Plot band structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--slab_dir", type=str, default=None,
            help="If slab, growth direction")
    parser.add_argument("--DFT_only", action='store_true',
            help="Use DFT bands only (no Wannier)")
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot, relative to E_F")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot, relative to E_F")
    parser.add_argument("--plot_evecs", action='store_true',
            help="Plot eigenvector decomposition")
    parser.add_argument("--show", action='store_true',
            help="Show band plot instead of outputting file")
    args = parser.parse_args()

    if args.slab_dir is None:
        # bulk calculation
        # assume SC10 path used
        band_path_syms, band_path_labels = sc10_fcc_path_syms()
    else:
        # slab calculation
        if args.slab_dir == "111":
            band_path_syms, band_path_labels = slab_fcc_111_path_syms()
        else:
            raise ValueError("unsupported slab_dir")

    work = get_work(args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    if args.minE is not None and args.maxE is not None:
        minE_plot = E_F + args.minE
        maxE_plot = E_F + args.maxE
    else:
        minE_plot, maxE_plot = None, None

    alat = alat_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)

    if args.DFT_only:
        Hr = None
    else:
        Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
        Hr = extractHr(Hr_path)

    bands_dir = os.path.join(work, "bands")
    bands_path = os.path.join(bands_dir, "{}_bands.dat".format(args.prefix))

    num_bands, num_ks, qe_bands = extractQEBands(bands_path)
    outpath = args.prefix

    plotBands(qe_bands, Hr, alat, latVecs, minE_plot, maxE_plot, outpath, show=args.show,
            symList=band_path_labels, fermi_energy=E_F, plot_evecs=args.plot_evecs)

if __name__ == "__main__":
    _main()
