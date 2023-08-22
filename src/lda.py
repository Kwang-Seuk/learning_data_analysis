# Loading modules
import os
import pandas as pd
import numpy as np
import math
from hyperopt import fmin, tpe
from typing import Union, Dict, Tuple
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from BorutaShap import BorutaShap
from xgboost import XGBRegressor, plot_tree
from xgboost import cv as xgb_cv


# Fuctions for "step1_prepare_modelling_data"
def load_and_check_data(
    data_dir: str, data_file: str, round_no: int = 3
) -> DataFrame:
    df = pd.read_csv(data_dir + data_file, index_col=0)

    # 1. Check for NA values and replace them with column mean
    df.fillna(df.mean(), inplace=True)

    # 2. Check for excessively long decimal places and round
    df = df.round(round_no)

    # 3. Check for columns with only one unique value (0 or any other value)
    for column in df.columns:
        if df[column].nunique() == 1:
            print(f"Warning: Column {column} has only one unique value.")
            print("It's recommended to remove it.")

    # 4. Check for non-numeric data
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Warning: Column {column} is not numeric.")
            print("Please check the values.")

    return df


def prepare_train_test_data(
    df: DataFrame, train_size: int, target_var: str
) -> DataFrame:
    # Check for the input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a DataFrame")

    if (
        not isinstance(train_size, int)
        or train_size <= 0
        or train_size >= len(df)
    ):
        raise ValueError(
            "train_size should be a positive integer, smaller than rows no. in df"
        )

    if target_var not in df.columns:
        raise ValueError("target_var should be one of the columns in df")

    # Data spliting
    df_model = df.copy()
    y = df_model[target_var]
    X = df_model.drop([target_var], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test


# Functions for "step2_develop_your_model"
def feat_selec_with_borutashap(
    x_train: DataFrame,
    x_test: DataFrame,
    y_train: Series,
    sel_trial: int,
    gpu_flag: bool = False,
) -> DataFrame:
    if gpu_flag is True:
        xgbm_pre = XGBRegressor(
            objective="reg:squarederror", tree_method="gpu_hist"
        )
    else:
        xgbm_pre = XGBRegressor(
            objective="reg:squarederror", tree_method="approx"
        )

    # Making a feature selection frame with BorataShap module
    Feature_Selector = BorutaShap(
        model=xgbm_pre,
        importance_measure="shap",
        classification=False,
        percentile=100,
        pvalue=0.05,
    )

    # Fitting and selecting input features
    Feature_Selector.fit(
        X=x_train,
        y=y_train,
        n_trials=sel_trial,
        sample=False,
        train_or_test="train",
        normalize=True,
        verbose=False,
    )

    # Create new input features sets for training and testing
    features_to_remove = Feature_Selector.features_to_remove
    x_train_sel = x_train.drop(columns=features_to_remove)
    x_test_sel = x_test.drop(columns=features_to_remove)

    # Sort input features by column titles
    x_train_sel = x_train_sel.reindex(sorted(x_train_sel.columns), axis=1)
    x_test_sel = x_test_sel.reindex(sorted(x_test_sel.columns), axis=1)

    print(
        "The number of selected input features is ",
        len(x_train_sel.columns),
        "including...",
        str(x_train_sel.columns),
    )

    return x_train_sel, x_test_sel


def hyper_parameters_objective(
    params: dict, x_train: DataFrame, y_train: Series, gpu_flag: bool = False
):
    tree_method = "gpu_hist" if gpu_flag else "auto"
    xgb_hpo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        max_depth=int(params["max_depth"]),
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        eta=params["eta"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        tree_method=tree_method,
        n_jobs=-1,
    )
    best_score = cross_val_score(
        xgb_hpo,
        x_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=10,
        error_score="raise",
    ).mean()
    loss = 1 - best_score
    return loss


def hyper_parameters_tuning(
    params: Dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_evals: int = 50,
    random_state: int = 777,
) -> Dict:
    best = fmin(
        fn=lambda p: hyper_parameters_objective(p, x_train, y_train),
        space=params,
        max_evals=max_evals,
        rstate=np.random.default_rng(random_state),
        algo=tpe.suggest,
    )

    assert isinstance(best, dict), "best must be a dictionary"

    print("The optimized hyper_parameters are: ")
    print(best)

    return best


def develop_production_model(
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    best: dict,
    iterations: int,
    gpu_flag: bool = False,
):
    x_train, y_train = train_data
    x_test, y_test = test_data

    best["max_depth"] = int(best["max_depth"])

    if gpu_flag is True:
        xgb_model_production = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=iterations,
            n_jobs=1,
            **best,
            tree_method="gpu_hist",
        )
    else:
        xgb_model_production = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=iterations,
            n_jobs=1,
            **best,
            tree_method="approx",
        )

    xgb_model_production.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric=["rmse"],
        verbose=100,
        early_stopping_rounds=15,
    )

    return xgb_model_production


def production_model_rmse_display(model):
    learning_res = model.evals_result()

    assert isinstance(learning_res, dict), "learning_res must be a dictionary"
    assert (
        "validation_0" in learning_res
    ), "learning_res should contain 'validation_0'"
    assert (
        "validation_1" in learning_res
    ), "learning_res should contain 'validation_1'"
    assert isinstance(
        learning_res["validation_0"]["rmse"], list
    ), "validation_0 rmse should be a list"
    assert isinstance(
        learning_res["validation_1"]["rmse"], list
    ), "validation_1 rmse should be a list"

    epochs = len(learning_res["validation_0"]["rmse"])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, learning_res["validation_0"]["rmse"], label="Train")
    ax.plot(x_axis, learning_res["validation_1"]["rmse"], label="Test")
    ax.legend()
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.title("XGBoost RMSE")
    plt.show()


# Functions for "step3_model_performance_test"
def best_tree_illustration(model: XGBRegressor, fig_dpi: int, fig_save: bool):
    plt.figure()
    plt.rcParams["figure.dpi"] = fig_dpi
    plot_tree(
        model,
        num_trees=model.get_booster().best_iteration,
    )
    plt.tight_layout()
    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(cwd + "/fig_test_tree.jpg", dpi=fig_dpi)
    plt.show()
    plt.close()


def predict_plot_train_test(
    model: XGBRegressor,
    train_data: Tuple[DataFrame, Series],
    test_data: Tuple[DataFrame, Series],
    fig_dpi: int = 300,
    fig_save: bool = False,
):
    plt.figure()
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train, y_train = train_data
    x_test, y_test = test_data

    tr_pred = model.predict(x_train)
    plt.scatter(y_train, tr_pred, label="Training data")

    tst_pred = model.predict(x_test)
    plt.scatter(y_test, tst_pred, label="Testing data")
    plt.xlabel("Observed y data")
    plt.ylabel("Predicted y data")
    plt.legend()
    plt.tight_layout()
    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_train_test_prediction_results.png", dpi=fig_dpi
        )
    plt.show()
    plt.close()


def feat_importance_general(
    model: XGBRegressor,
    train_data: Tuple[DataFrame, Series],
    fig_dpi: int = 300,
    fig_save: bool = False,
):
    plt.figure()
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train, x_test = train_data
    feature_names = np.array(x_train.columns)
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.xlabel("XGBoost Feature Importance")
    plt.tight_layout()
    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(cwd + "/fig_feature_importance.png", dpi=fig_dpi)
    plt.show()
    plt.close()


def feat_importance_permut(
    model: XGBRegressor,
    train_data: Tuple[DataFrame, Series],
    fig_dpi: int = 300,
    fig_save: bool = False,
):
    plt.figure()
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train, y_train = train_data

    perm_importance = permutation_importance(model, x_train, y_train)
    feature_names = np.array(x_train.columns)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(
        feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
    )
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_feature_importance_permuted.png",
            dpi=fig_dpi,
        )
    plt.show()
    plt.close()


# Most-influencing parameter
def create_interpolation_for_mip(
    x_train: DataFrame, interval: int
) -> DataFrame:
    assert isinstance(interval, int), "interval must be an integer"

    minmax_df = pd.DataFrame(x_train.nunique(), columns=["nunique"])
    minmax_df["min"] = x_train.min()
    minmax_df["max"] = x_train.max()
    minmax_df["dtypes"] = x_train.dtypes

    df_interpolated = pd.DataFrame()
    for _, col_name in enumerate(minmax_df.index):
        temp_linspace = np.linspace(
            minmax_df["min"][col_name], minmax_df["max"][col_name], interval
        )
        df_interpolated[col_name] = temp_linspace

    return df_interpolated


def create_df_means_for_mip(
    x_train: DataFrame, df_interpolated: DataFrame
) -> DataFrame:
    df_mean_tmp = pd.DataFrame(x_train.mean(), columns=["mean"])
    df_means = pd.DataFrame(
        columns=list(x_train.columns), index=range(len(df_interpolated))
    )
    for col_name in df_mean_tmp.index:
        df_means[col_name] = df_mean_tmp["mean"][col_name]

    return df_means


def mip_analysis(
    df_means: DataFrame,
    df_itp: DataFrame,
    model: XGBRegressor,
    save_res: bool = True,
):
    # store data for further plot creation
    df_mip_res = pd.DataFrame()
    df_means_copy = df_means.copy()

    for col_name in df_means.columns:
        df_means_copy[col_name] = df_itp[col_name]
        mip_pred = model.predict(df_means_copy)
        df_mip_res[col_name] = mip_pred.tolist()

    # save the mip results as a csv file
    if save_res is True:
        cwd = os.getcwd()
        df_mip_res.to_csv(cwd + "/mip_analysis_res" + ".csv")

    return df_mip_res


def plot_mip_analysis_results(
    df_itp: DataFrame,
    df_mip_res: DataFrame,
    fig_dpi: int,
    fig_size: tuple,
    fig_save: bool = True,
):
    plt.figure()
    col_list = list(df_itp.columns)

    fig = plt.figure(figsize=fig_size)
    for i, col_name in enumerate(col_list):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.plot(df_itp[col_name], df_mip_res[col_name], "r-")
        ax.set_title(col_name)

        # Check if it's the first column
        if i % 5 == 0:
            ax.set_ylabel("Predicted result")

        fig.tight_layout()

    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_mip_results.png",
            dpi=fig_dpi,
        )
    plt.show()
    plt.close()


# randomized dataframes creation
def create_random_dataframes(df_interpolated: DataFrame, n: int) -> None:
    interval = df_interpolated.shape[0]

    df_rand = pd.DataFrame(
        {
            column: np.random.uniform(
                df_interpolated[column].min(), df_interpolated[column].max(), n
            )
            for column in df_interpolated.columns
        }
    )
    df_rand_tmp = pd.concat([df_rand] * interval, ignore_index=True)
    df_interval_tmp = pd.DataFrame(
        {
            column: np.repeat(df_interpolated[column].values, n)
            for column in df_interpolated.columns
        }
    )
    df_frames = [
        df_rand_tmp.copy().assign(**{column: df_interval_tmp[column]})
        for column in df_interpolated.columns
    ]
    for i, df in enumerate(df_frames):
        filename = f"df_randomized_{df_interpolated.columns[i]}.csv"
        df.to_csv(filename, index=False)


# Random simulation
def random_simulation(
    input_csv: str, model: Union[XGBRegressor, str], save_res: bool = True
) -> DataFrame:
    df_rand = pd.read_csv(input_csv)

    if isinstance(model, str):
        loaded_model = XGBRegressor()
        loaded_model.load_model(model)
        model = loaded_model

    rand_pred = model.predict(df_rand)
    df_rand["y"] = rand_pred.tolist()

    if save_res is True:
        cwd = os.getcwd()
        base = os.path.basename(input_csv)
        filename = os.path.splitext(base)[0]
        df_rand.to_csv(
            cwd + "/random_simulation_res_" + filename + ".csv", index=False
        )

    return df_rand


def draw_scatter_graphs_from_csv(
    csv_file: str, control_var: str, x_var: str, y_var: str
) -> None:
    plt.figure()
    df = pd.read_csv(csv_file)

    interval = df[control_var].nunique()

    # Calculate subplot grid dimensions
    grid_size = math.ceil(math.sqrt(interval))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.ravel()  # Flatten axis array

    # Determine dot size based on dataframe size
    dot_size = max(5, 5000 / len(df))

    for subplot_index, unique_val in enumerate(df[control_var].unique()):
        subset = df[df[control_var] == unique_val]
        scatter = axs[subplot_index].scatter(
            subset[x_var],
            subset[y_var],
            c=subset["y"],
            cmap="jet",
            s=dot_size,
        )
        fig.colorbar(scatter, ax=axs[subplot_index])
        axs[subplot_index].set_title(f"{control_var} = {unique_val}")

        if (subplot_index + grid_size) // grid_size >= grid_size:
            axs[subplot_index].set_xlabel(x_var)

        if subplot_index % grid_size == 0:
            axs[subplot_index].set_ylabel(y_var)

    # Remove unused subplots
    for i in range(interval, grid_size * grid_size):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    plt.close()


def draw_contour_graphs_from_csv(
    csv_file: str, control_var: str, x_var: str, y_var: str
) -> None:
    plt.figure()
    df = pd.read_csv(csv_file)
    interval = df[control_var].nunique()

    # Calculate subplot grid dimensions
    grid_size = math.ceil(math.sqrt(interval))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.ravel()  # Flatten axis array

    for subplot_index, unique_val in enumerate(df[control_var].unique()):
        subset = df[df[control_var] == unique_val]
        contour = axs[subplot_index].tricontourf(
            subset[x_var], subset[y_var], subset["y"], cmap="jet"
        )
        fig.colorbar(contour, ax=axs[subplot_index])
        axs[subplot_index].set_title(f"{control_var} = {unique_val}")

        if (
            subplot_index // grid_size == grid_size - 1
        ):  # Only label x-axis on the bottom-most subplots
            axs[subplot_index].set_xlabel(x_var)

        if (
            subplot_index % grid_size == 0
        ):  # Only label y-axis on the left-most subplots
            axs[subplot_index].set_ylabel(y_var)

    # Remove unused subplots
    for i in range(interval, grid_size * grid_size):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    plt.close()
