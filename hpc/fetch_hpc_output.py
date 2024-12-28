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
    for exp in experiment_ids:
        local_exp_path = f"/Users/{user}/Documents/ITU/master_thesis/hpc/jobs/{exp}"
        if not os.path.exists(f"{local_exp_path}"):
            os.makedirs(f"{local_exp_path}")

        get_experiment_json(exp, f"{local_exp_path}")


def fetch_predictions(experiment_id: str, fold: int = 0, user: str = "newuser") -> None:
    """
    Fetch the model weights from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/out/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
    out_dir = (
        f"/Users/{user}/Documents/ITU/master_thesis/out/predictions/{experiment_id}"
    )
    if not os.path.exists(out_dir):
        print("Creating output directory")
        os.makedirs(out_dir, exist_ok=True)
    _transfer_hpc_to_local(hpc_out, out_dir)


def fetch_benchmark(user: str = "newuser") -> None:
    """
    Fetch the model weights from the HPC
    """
    hpc_out = f"maxd@hpc.itu.dk:~/master_thesis/model/model_benchmark_results.json"
    out_dir = (
        f"/Users/{user}/Documents/ITU/master_thesis/model/model_benchmark_results.json"
    )
    _transfer_hpc_to_local(hpc_out, out_dir)


# %%
if __name__ == "__main__":
    # TODO clean this up

    from utils.data_models import ExperimentAnalysis

    # # SCRIPT PARAMS ---------
    # with open("experiment_analysis_parameters.json", "r") as f:
    #     config = ExperimentAnalysis.model_validate(json.load(f))
    # experiment_id = config.experiment_id
    # job_id = config.hpc_job_id
    experiment_id = "c70_3D_2411_1824"

    # local_user = "newuser"
    # local_exp_path = (
    #     f"/Users/{local_user}/Documents/ITU/master_thesis/hpc/jobs/{experiment_id}"
    # )

    # if not os.path.exists(local_exp_path):
    #     os.makedirs(local_exp_path)

    # Experiment level data
    # get_job_stdout(job_id, local_exp_path)
    # get_experiment_json(experiment_id, local_exp_path)

    # Fold level data
    # folds = [0, 1, 2, 3, 4]
    # for f in folds:
    #     fold_path = f"{local_exp_path}/fold_{f}"
    #     if not os.path.exists(fold_path):
    #         os.makedirs(fold_path)
    #     get_fold_json(experiment_id, f, fold_path)
    #     update_loss_plot(experiment_id, f, local_user)
    #     update_error_distribution(experiment_id, f, local_user)
    #     fetch_model_weights(experiment_id, f, local_user)
    #     fetch_predictions(experiment_id, f, local_user)

    # FETCH RESULTS FOR TREND PLOT
    # experiments = [  # 2.5D
    #     "c40_25D_2811_2153",
    #     "c50_25D_2811_2112",
    #     "c60_25D_2811_2111",
    #     "c70_25D_2811_2106",
    # ]
    # fetch_all_final_experiment_results(experiments, user="newuser")

    # %%
