import argparse
from gpaw_run_util.post_soc import run_soc_ev

# NOTE: Run this in serial only - get_spinorbit_eigenvalues is broken for parallel.
# It assumes all k-points are on one process.

def _main():
    parser = argparse.ArgumentParser("Get and emit information for bands")
    parser.add_argument("prefix", type=str, help="System prefix")
    args = parser.parse_args()

    run_soc_ev(args.prefix)

if __name__ == "__main__":
    _main()
