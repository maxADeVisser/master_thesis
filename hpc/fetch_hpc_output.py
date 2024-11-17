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
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/job.{job_id}.out"
    _transfer_hpc_to_local(hpc_out, dest_dir)


def get_experiment_json(experiment_id: str, dest_dir: str) -> None:
    """
    Fetch the final experiment json file from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/run_{experiment_id}.json"
    _transfer_hpc_to_local(hpc_out, dest_dir)


def get_fold_json(experiment_id: str, fold: int, dest_dir: str) -> None:
    """
    Fetch the final experiment json file from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/fold{fold}/fold{fold}_{experiment_id}.json"
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
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/model_runs/{experiment_id}/fold{fold}/model.pth"
    out_dir = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}/fold_{fold}"
    _transfer_hpc_to_local(hpc_out, out_dir)


# %%
if __name__ == "__main__":
    experiment_id = "c30_3D_1711_1513"
    job_id = 1393
    local_user = "maxvisser"
    local_exp_path = (
        f"/Users/{local_user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    )

    if not os.path.exists(local_exp_path):
        os.makedirs(local_exp_path)

    # Experiment level data
    get_job_stdout(job_id, local_exp_path)
    get_experiment_json(experiment_id, local_exp_path)

    # Fold level data
    # folds = [0, 1, 2, 3, 4]
    folds = [2, 3]
    for f in folds:
        fold_path = f"{local_exp_path}/fold_{f}"
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        get_fold_json(experiment_id, f, fold_path)
        update_loss_plot(experiment_id, f, local_user)
        # update_error_distribution(experiment_id, f, local_user)
        # fetch_model_weights(experiment_id, fold, local_user)
