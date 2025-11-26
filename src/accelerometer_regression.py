import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def process():
    """Process accelerometer data for regression analysis"""
    print("Loading accelerometer data...")

    # Read the data
    df = pd.read_csv("data/clean_accelerometer.csv")

    # Separate recorded and calculated data
    recorded_df = df[df["data type"] == "recorded"]
    calculated_df = df[df["data type"] == "calculated"]

    # For each dimension, create a scatter plot comparing recorded vs calculated values
    dimensions = ["x", "y", "z"]

    for dimension in dimensions:
        print(f"Processing {dimension} dimension...")

        # Filter data for current dimension
        recorded_dim = recorded_df[recorded_df["dimension"] == dimension]
        calculated_dim = calculated_df[calculated_df["dimension"] == dimension]

        # Since angles may not exactly match, we need to merge on angle
        # Let's group by angle and take mean values to ensure one-to-one mapping
        recorded_agg = recorded_dim.groupby("angle")["value"].mean().reset_index()
        calculated_agg = calculated_dim.groupby("angle")["value"].mean().reset_index()

        # Merge on angle
        merged_df = pd.merge(
            recorded_agg,
            calculated_agg,
            on="angle",
            suffixes=("_recorded", "_calculated"),
        )

        # Create scatter plot
        plt.figure()
        sns.scatterplot(data=merged_df, x="value_recorded", y="value_calculated")
        plt.xlabel(f"Recorded {dimension} values")
        plt.ylabel(f"Calculated {dimension} values")
        plt.title(f"Recorded vs Calculated {dimension.upper()} Values")

        # Perform linear regression
        X = merged_df[["value_recorded"]]
        y = merged_df["value_calculated"]
        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)

        # Plot regression line using seaborn's lineplot with dotted style
        # To use lineplot, we need to sort by x values to get a proper line
        plot_df = merged_df.copy()
        plot_df = plot_df.sort_values('value_recorded')
        plot_df['y_pred'] = regressor.predict(plot_df[['value_recorded']])
        sns.lineplot(data=plot_df, x='value_recorded', y='y_pred', 
                     linestyle='--', linewidth=2)

        # Add R² and regression coefficients to the plot
        r2 = r2_score(y, y_pred)
        slope = regressor.coef_[0]
        intercept = regressor.intercept_
        plt.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", alpha=0.5),
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"output/accelerometer_regression_{dimension}.png", dpi=150)
        plt.close()

        print(f"  R² score for {dimension}: {r2:.4f}")
        print(
            f"  Regression coefficients: {regressor.coef_[0]:.4f} (slope), {regressor.intercept_:.4f} (intercept)"
        )

    print("Regression analysis complete!")
