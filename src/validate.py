import os
from typing import Union
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd
from xgboost import XGBRegressor


def validate_train_test_data(
    x_train: DataFrame, x_test: DataFrame, y_train: Series, y_test: Series
):
    assert isinstance(x_train, DataFrame), "x_train should be a DataFrame"
    assert isinstance(x_test, DataFrame), "x_test should be a DataFrame"
    assert isinstance(y_train, Series), "y_train should be a Series"
    assert isinstance(y_test, Series), "y_test should be a Series"

    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "x_train and y_train should have the same number of rows"
    assert (
        x_test.shape[0] == y_test.shape[0]
    ), "x_test and y_test should have the same number of rows"

    # ensure that column names do not contain spaces or special characters
    invalid_columns = [
        col for col in x_train.columns if not col.isidentifier()
    ]
    assert (
        len(invalid_columns) == 0
    ), f"Columns {invalid_columns} contain invalid characters"
    # ensure that x_train and x_test have the same columns in the same order
    assert (
        x_train.columns == x_test.columns
    ).all(), (
        "x_train and x_test should have the same columns in the same order"
    )


def validate_selected_features(x_train_sel: DataFrame, x_test_sel: DataFrame):
    assert isinstance(
        x_train_sel, DataFrame
    ), "x_train_sel must be a pandas DataFrame"
    assert isinstance(
        x_test_sel, DataFrame
    ), "x_test_sel must be a pandas DataFrame"


def validate_production_model(model: XGBRegressor) -> None:
    assert isinstance(
        model, XGBRegressor
    ), "xgb_model_production must be an instance of XGBRegressor"


def validate_df_interpolated(
    df_interpolated: DataFrame, x_train: DataFrame, interval: int
):
    # Validate the output is a DataFrame
    assert isinstance(
        df_interpolated, DataFrame
    ), "Output should be a pandas DataFrame."

    # Validate the shape of the DataFrame
    assert df_interpolated.shape == (
        interval,
        x_train.shape[1],
    ), f"Output shape should be ({interval}, {x_train.shape[1]})."

    # Validate that column names match the input data
    assert (
        df_interpolated.columns.tolist() == x_train.columns.tolist()
    ), "Output column names should match the input data."

    # Validate that the values are interpolated correctly
    for column in df_interpolated.columns:
        assert (
            df_interpolated[column].min() == x_train[column].min()
        ), f"Minimum value for {column} should match the input data."
        assert (
            df_interpolated[column].max() == x_train[column].max()
        ), f"Maximum value for {column} should match the input data."
        assert (
            len(df_interpolated[column].unique()) == interval
        ), f"The number of unique values in {column} should be equal to the specified interval."


def validate_df_means(
    df_means: DataFrame, x_train: DataFrame, df_interpolated: DataFrame
):
    # Validate the output is a DataFrame
    assert isinstance(
        df_means, DataFrame
    ), "Output should be a pandas DataFrame."

    # Validate the shape of the DataFrame
    assert (
        df_means.shape == df_interpolated.shape
    ), "Output shape should match df_interpolated shape."

    # Validate that column names match the input data
    assert (
        df_means.columns.tolist() == x_train.columns.tolist()
    ), "Output column names should match the input data."

    # Validate that all values in each column are equal to the mean of
    # the corresponding column in the input data
    for column in df_means.columns:
        assert (
            df_means[column].nunique() == 1
        ), f"Column {column} should only contain one unique value."
        assert (
            df_means[column].unique()[0] == x_train[column].mean()
        ), f"Unique value in column {column} should be the mean of the corresponding column in the input data."


def validate_mip_analysis(
    df_mip_res: DataFrame, df_means: DataFrame, df_itp: DataFrame
):
    # Validate the output is a DataFrame
    assert isinstance(
        df_mip_res, DataFrame
    ), "Output should be a pandas DataFrame."

    # Validate the shape of the DataFrame
    assert (
        df_mip_res.shape[0] == df_itp.shape[0]
    ), "Output rows should match the number of rows in df_itp."
    assert (
        df_mip_res.shape[1] == df_means.shape[1]
    ), "Output columns should match the number of columns in df_means."

    # Validate that column names match the input data
    assert (
        df_mip_res.columns.tolist() == df_means.columns.tolist()
    ), "Output column names should match the input data."


def validate_create_random_dataframes(
    df_interpolated: DataFrame, n: int
) -> None:
    interval = df_interpolated.shape[0]

    # Check dataframe shape
    assert df_interpolated.shape == (interval, len(df_interpolated.columns))

    # Check dataframe values are within expected range
    for column in df_interpolated.columns:
        min_val = df_interpolated[column].min()
        max_val = df_interpolated[column].max()
        assert df_interpolated[column].between(min_val, max_val).all()

    # Check saved CSV files
    for i in range(len(df_interpolated.columns)):
        filename = f"df_randomized_{df_interpolated.columns[i]}.csv"
        assert os.path.isfile(filename)

        # Check CSV file content
        df_saved = pd.read_csv(filename)
        assert df_saved.shape == (interval * n, len(df_interpolated.columns))

        # Check dataframe values are within expected range
        for column in df_interpolated.columns:
            min_val = df_interpolated[column].min()
            max_val = df_interpolated[column].max()
            assert df_saved[column].between(min_val, max_val).all()

    print("All tests passed.")


def validate_model_loading(model: Union[XGBRegressor, str]) -> None:
    if isinstance(model, str):
        loaded_model = XGBRegressor()
        try:
            loaded_model.load_model(model)
            print("Model loading test passed.")
        except Exception as e:
            print(f"Model loading failed with error: {e}")
    else:
        print("Input is not a model path.")


def validate_saved_csv(input_csv, df_input):
    cwd = os.getcwd()
    filename = os.path.splitext(os.path.basename(input_csv))[0]
    df_saved = pd.read_csv(cwd + "/random_simulation_res_" + filename + ".csv")
    assert df_saved.shape[0] == len(
        df_input
    ), "Saved CSV file row number mismatch."
    assert df_saved.shape[1] == len(
        df_input.columns
    ), "Saved CSV file column number mismatch."
    assert (
        "y" in df_saved.columns
    ), "Predicted 'y' column not found in saved CSV file."
    print("Saved CSV file test passed.")
