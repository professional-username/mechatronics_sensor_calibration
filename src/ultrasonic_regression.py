import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_and_prepare_data(file_path):
    """Load ultrasonic data"""
    print("Loading ultrasonic data...")
    df = pd.read_csv(file_path)
    return df


def perform_linear_regression(df):
    """Perform linear regression on value vs distance"""
    X = df[["distance"]]
    y = df["value"]
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    return regressor, r2, y_pred


def create_regression_plot(df, regressor, r2, surface):
    """Create and save regression plot"""
    plt.figure()
    sns.scatterplot(data=df, x="distance", y="value")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.title(f"Ultrasonic Sensor Calibration (Surface: {surface})")

    # Prepare data for regression line
    plot_df = df.copy()
    plot_df = plot_df.sort_values("distance")
    plot_df["value_pred"] = regressor.predict(plot_df[["distance"]])

    # Plot regression line
    sns.lineplot(
        data=plot_df, x="distance", y="value_pred", linestyle="--", linewidth=2
    )

    # Add regression metrics to the plot
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    plt.text(
        0.05,
        0.90,
        f"R² = {r2:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        f"output/plots/ultrasonic/ultrasonic_regression_surface_{surface}.png", dpi=150
    )
    plt.close()


def create_lookup_table(df, regressor, surface):
    """Create distance->value lookup table using linear regression"""
    # Get unique distances from the data
    unique_distances = np.sort(df["distance"].unique())

    # Predict value using the regression model
    lookup_data = []
    for distance in unique_distances:
        value = regressor.predict([[distance]])[0]
        lookup_data.append({"distance": distance, "value": value})

    lookup_df = pd.DataFrame(lookup_data)
    return lookup_df


def process():
    """Main function to process ultrasonic data for regression analysis"""
    # Load data
    df = load_and_prepare_data("data/clean_ultrasonic.csv")

    # Get all unique surfaces
    surfaces = df["surface"].unique()
    print(f"Found surfaces: {list(surfaces)}")

    for surface in surfaces:
        print(f"\nProcessing surface: {surface}")
        # Filter data for current surface
        filtered_df = df[df["surface"] == surface]
        print(f"Data points: {len(filtered_df)}")

        # Perform regression
        regressor, r2, y_pred = perform_linear_regression(filtered_df)

        print(f"R² score: {r2:.4f}")
        print(
            f"Regression coefficients: {regressor.coef_[0]:.4f} (slope), {regressor.intercept_:.4f} (intercept)"
        )

        # Create plot
        create_regression_plot(filtered_df, regressor, r2, surface)

        # Create lookup table
        print("Creating lookup table...")
        lookup_df = create_lookup_table(filtered_df, regressor, surface)
        filename = f"output/lookup_tables/ultrasonic/ultrasonic_lookup_table_surface_{surface}.csv"
        lookup_df.to_csv(filename, index=False)
        print(f"Lookup table saved to {filename}")
        print(f"Table shape: {lookup_df.shape}")
