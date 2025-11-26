import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Global configuration variables
DIMENSIONS_TO_PLOT = ["x", "z"]
RECORDED_DATA_PLOT = "scatter"  # Options: "scatter", "boxplot"


def process():
    # Load data
    df = pd.read_csv("data/clean_accelerometer.csv")
    print(f"Loaded accelerometer data with shape: {df.shape}")

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Separate recorded and calculated data
    recorded = df[df["data type"] == "recorded"]
    calculated = df[df["data type"] == "calculated"]

    # Filter dimensions to plot
    recorded = recorded[recorded["dimension"].isin(DIMENSIONS_TO_PLOT)]
    calculated = calculated[calculated["dimension"].isin(DIMENSIONS_TO_PLOT)]

    # Plot recorded data based on the selected plot type
    if RECORDED_DATA_PLOT == "scatter":
        for dim in DIMENSIONS_TO_PLOT:
            dim_data = recorded[recorded["dimension"] == dim]
            sns.scatterplot(
                data=dim_data, x="angle", y="value", label=f"{dim} recorded", alpha=0.6
            )
    elif RECORDED_DATA_PLOT == "boxplot":
        # Create boxplot for each angle
        # We need to ensure the x-axis is treated as categorical for boxplot
        # To avoid too many boxes, we can use the actual angle values
        # Since angles are from 0-360, and there are 6365/3 ~= 2121 points per dimension, which is manageable
        for dim in DIMENSIONS_TO_PLOT:
            dim_data = recorded[recorded["dimension"] == dim]
            # Sort by angle to ensure proper ordering
            dim_data = dim_data.sort_values("angle")
            sns.boxplot(data=dim_data, x="angle", y="value", label=f"{dim} recorded")
            # Note: Boxplot with many categories may be very dense
            # We might need to adjust the visualization if there are too many angles

    # Plot calculated data as line plot
    for dim in DIMENSIONS_TO_PLOT:
        dim_data = calculated[calculated["dimension"] == dim]
        sns.lineplot(
            data=dim_data,
            x="angle",
            y="value",
            label=f"{dim} calculated",
            linestyle="--",
        )

    plt.title("Accelerometer Data by Angle and Dimension")
    plt.xlabel("Angle")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/accelerometer_plot.png")
    plt.close()

    print("Accelerometer visualization saved to output/accelerometer_plot.png")
