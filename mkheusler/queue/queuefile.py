import os
import stat
from mkheusler.queue.queue_util import _global_config, _base_dir

def write_queuefile(config):
    machine = config["machine"]

    gconf = _global_config()

    if machine == "ls5" or machine == "stampede2":
        _write_queuefile_tacc(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def _write_queuefile_tacc(config):
    duration = _ls_format_duration(config["hours"], config["minutes"])
    prefix = config["prefix"]

    qf = ["#!/bin/bash"]
    qf.append("#SBATCH -p {}".format(config["queue"]))
    qf.append("#SBATCH -J {}".format(prefix))
    qf.append("#SBATCH -o {}.out".format(prefix))
    qf.append("#SBATCH -e {}.err".format(prefix))
    qf.append("#SBATCH -t {}".format(duration))
    qf.append("#SBATCH -N {}".format(str(config["nodes"])))
    qf.append("#SBATCH -n {}".format(str(config["mpi_tasks"])))
    qf.append("#SBATCH -A {}".format(config["project"]))
    qf.append("")
    qf.append("export OMP_NUM_THREADS={}".format(str(config["openmp_threads_per_mpi_task"])))

    if config["calc"] == "wan_setup":
        nk = str(config["qe_pools"])
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.scf.in > scf.out".format(nk, prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.bands.in > bands.out".format(nk, prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.nscf.in > nscf.out".format(nk, prefix))
    elif config["calc"] == "pw_post":
        qf.append("cd ../bands")
        qf.append("ibrun tacc_affinity bands.x -input {}.bands_post.in > bands_post.out".format(prefix))
        qf.append("rm {}.wfc*".format(prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("wannier90.x -pp {}".format(prefix))
            qf.append("ibrun tacc_affinity pw2wannier90.x -input {}.pw2wan.in > pw2wan.out".format(prefix))
            # Clean up redundant wfc extracted from .save directory
            qf.append("rm {}.wfc*".format(prefix))
            # Go ahead and run W90: assume windows set as needed.
            qf.extend(_wan_run(config))
    elif config["calc"] == "wan_run":
        qf.extend(_wan_run(config))
    else:
        raise ValueError("unrecognized config['calc'] ('wan_setup' and 'wan_run' supported)")

    qf_path = get_qf_path(config)

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf) + "\n"
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)

def _wan_run(config):
    wan_dir = os.path.join(config["base_path"], config["prefix"], "wannier")
    update_dis = os.path.join(_base_dir(), "mkheusler", "wannier", "update_dis.py")
    outer_min, outer_max = str(config["outer_min"]), str(config["outer_max"])
    inner_min, inner_max = str(config["inner_min"]), str(config["inner_max"])
    py_str = "python3 '{}' '{}' {} {} {} {}".format(update_dis, config["prefix"], outer_min, outer_max, inner_min, inner_max)
    if "subdir" in config and config["subdir"] is not None:
        py_str = py_str + " --subdir {}".format(config["subdir"])

    qf = [py_str]
    qf.append("wannier90.x {}".format(config["prefix"]))

    return qf

def _ls_format_duration(hours, minutes):
    hstr = str(hours)
    if minutes < 10:
        mstr = "0{}".format(str(minutes))
    else:
        mstr = str(minutes)

    return "{}:{}:00".format(hstr, mstr)

def get_qf_path(config):
    qf_name = "run_{}".format(config["calc"])
    qf_path = os.path.join(config["base_path"], config["prefix"], "wannier", qf_name)

    return qf_path
