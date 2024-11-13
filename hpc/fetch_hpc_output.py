# %%
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


def update_error_distribution(
    experiment_id: str, fold: int, user: str = "newuser"
) -> None:
    """
    Fetch the latest error distribution from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/fold{fold}/error_distribution.png"
    out_dir = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}/fold_{fold}"
    _transfer_hpc_to_local(hpc_out, out_dir)


def update_loss_plot(experiment_id: str, fold: int, user: str = "newuser") -> None:
    """
    Fetch the latest loss plot from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/fold{fold}/loss_plot.png"
    out_dir = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}/fold_{fold}"
    _transfer_hpc_to_local(hpc_out, out_dir)


def fetch_model_weights(experiment_id: str, fold: int, user: str = "newuser") -> None:
    """
    Fetch the model weights from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/fold{fold}/model_fold{fold}.pth"
    out_dir = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}/fold_{fold}"
    _transfer_hpc_to_local(hpc_out, out_dir)


# %%
if __name__ == "__main__":
    experiment_id = "c50_25d_1311_1450"
    job_id = 784
    fold = 0
    local_user = "maxvisser"  # maxvisser for personal computer

    # Create out folder
    local_path = f"/Users/{local_user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}/fold_{fold}"
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    get_job_stdout(job_id, local_path)
    update_loss_plot(experiment_id, fold, local_user)
    update_error_distribution(experiment_id, fold, local_user)
    # fetch_model_weights(experiment_id, fold, local_user)
