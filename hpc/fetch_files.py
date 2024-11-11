import os
import subprocess

from project_config import env_config


def sync_output_files(
    experiment_id: str, server_path="maxd@hpc.itu.dk:~/master_thesis"
) -> None:
    """
    Fetch the latest data from the HPC and update the plots for a experiment.
    see the experiment_ids in the `out` directory on the HPC.
    """
    out_dir = "out"
    print("Fetching new data from the HPC ...")
    try:
        os.chdir(out_dir)
        files = os.listdir()
        subprocess.run(["scp", f"{server_path}/out", f"{env_config}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
