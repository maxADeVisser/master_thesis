# %%
import json
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


def fetch_all_final_experiment_results(
    experiment_ids: list[str], user: str = "newuser"
) -> None:
    """Fetch all the final results from the HPC for each experiment in the list

    Args:
        experiment_ids (list[str]): List of experiment ids to fetch
        user (str, optional): _description_. Defaults to "newuser".
    """
    for e in experiment_ids:
        local_exp_path = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{e}"
        if not os.path.exists(f"{local_exp_path}/{e}"):
            os.makedirs(f"{local_exp_path}/{e}")

        get_experiment_json(e, f"{local_exp_path}")


# %%
if __name__ == "__main__":
    # TODO clean this up

    from utils.data_models import ExperimentAnalysis

    # SCRIPT PARAMS ---------
    with open("experiment_analysis_parameters.json", "r") as f:
        config = ExperimentAnalysis.model_validate(json.load(f))

    experiment_id = config.experiment_id
    job_id = config.hpc_job_id

    local_user = "newuser"
    local_exp_path = (
        f"/Users/{local_user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    )

    if not os.path.exists(local_exp_path):
        os.makedirs(local_exp_path)

    # Experiment level data
    get_job_stdout(job_id, local_exp_path)
    get_experiment_json(experiment_id, local_exp_path)

    # Fold level data
    folds = config.analysis.folds
    for f in folds:
        fold_path = f"{local_exp_path}/fold_{f}"
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        get_fold_json(experiment_id, f, fold_path)
        update_loss_plot(experiment_id, f, local_user)
        update_error_distribution(experiment_id, f, local_user)
        fetch_model_weights(experiment_id, f, local_user)

    # FETCH RESULTS FOR TREND PLOT
    # experiments = [
    #     "c30_3D_1711_1513",
    #     "c50_3D_1711_2149",
    #     "c30_25D_1911_0928",
    #     "c50_25D_1911_1125",
    #     "c70_25D_1911_1411",
    # ]
    # fetch_all_final_experiment_results(experiments)

    # %%
