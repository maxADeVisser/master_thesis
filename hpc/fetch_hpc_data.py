import os
import subprocess


def _transfer_hpc_to_local(hpc_path: str, local_path: str) -> None:
    """
    Transfer data from the HPC to the local machine
    """
    try:
        subprocess.run(["scp", f"{hpc_path}", f"{local_path}"], check=True)
        print("Data fetched successfully (Y)")
    except subprocess.CalledProcessError as _:
        print(f"There was an error (maybe the file is not there)")


def get_job_stdout(job_id: int, dest_dir: str) -> None:
    """
    Fetch the latest stdout data from the HPC
    see the experiment_ids in the `out` directory on the HPC.
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/job.{job_id}.out"
    _transfer_hpc_to_local(hpc_out, dest_dir)


def update_loss_plot(experiment_id: str) -> None:
    """
    Fetch the latest loss plot from the HPC
    """
    hpc_out = (
        f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/loss_plot.png"
    )
    out_dir = f"/Users/maxvisser/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    _transfer_hpc_to_local(hpc_out, out_dir)


def fetch_model_weights(experiment_id: str) -> None:
    """
    Fetch the model weights from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/model_fold_0.pth"
    out_dir = f"/Users/maxvisser/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    _transfer_hpc_to_local(hpc_out, out_dir)


if __name__ == "__main__":
    experiment_id = "context30_1211_1114"
    job_id = 462

    # Create out folder
    path = f"/Users/maxvisser/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    if not os.path.exists(path):
        os.makedirs(path)

    get_job_stdout(job_id, path)
    update_loss_plot(experiment_id)
