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
        for i, dim in enumerate(DIMENSIONS_TO_PLOT):
            dim_data = recorded[recorded["dimension"] == dim]
            sns.scatterplot(
                data=dim_data,
                x="angle",
                y="value",
                label=f"{dim} recorded",
                alpha=0.6,
            )
    elif RECORDED_DATA_PLOT == "boxplot":
        # Use hue parameter to let seaborn handle colors automatically
        sns.boxplot(data=recorded, x="angle", y="value", hue="dimension")

    # Plot calculated data as line plot
    for i, dim in enumerate(DIMENSIONS_TO_PLOT):
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
