import subprocess


def get_job_stdout(job_id: int, dest_dir: str) -> None:
    """
    Fetch the latest stdout data from the HPC
    see the experiment_ids in the `out` directory on the HPC.
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/job.{job_id}.out"
    try:
        subprocess.run(["scp", f"{hpc_out}", f"{dest_dir}"], check=True)
        print("Data fetched successfully (Y)")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def update_loss_plot(experiment_id: int, dest_dir: str) -> None:
    """
    Fetch the latest loss plot from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/"
    try:
        subprocess.run(["scp", f"{hpc_out}", f"{dest_dir}"], check=True)
        print("Data fetched successfully (Y)")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def fetch_model_weights(job_id: int, dest_dir: str) -> None:
    """
    Fetch the model weights from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs"
    try:
        subprocess.run(["scp", f"{hpc_out}", f"{dest_dir}"], check=True)
        print("Data fetched successfully (Y)")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_job_stdout(418, "/Users/maxvisser/Documents/ITU/master_thesis/hpc/job_files")
