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
        # Use hue parameter for consistent colors
        sns.scatterplot(
            data=recorded,
            x="angle",
            y="value",
            hue="dimension",
            style="dimension",
            alpha=0.6,
        )
    elif RECORDED_DATA_PLOT == "boxplot":
        # Use hue parameter to let seaborn handle colors automatically
        sns.boxplot(data=recorded, x="angle", y="value", hue="dimension")

    # Plot calculated data as line plot using hue for consistent colors
    # We need to differentiate from recorded data, so we'll use linestyle
    # Create a custom style mapping
    calculated_with_style = calculated.copy()
    calculated_with_style['plot_type'] = 'calculated'
    
    sns.lineplot(
        data=calculated_with_style,
        x="angle",
        y="value",
        hue="dimension",
        style="plot_type",
        dashes=[(2, 2)],  # Dashed line for all calculated data
        markers=False,
    )

    plt.title("Accelerometer Data by Angle and Dimension")
    plt.xlabel("Angle")
    plt.ylabel("Value")
    
    # Get the current legend and modify it to remove 'plot_type' entries
    legend = plt.gca().get_legend()
    if legend:
        # Filter out legend entries that contain 'plot_type' information
        # We want to keep only dimension labels
        handles, labels = [], []
        for handle, label in zip(legend.legend_handles, legend.get_texts()):
            label_text = label.get_text()
            # Only add entries that are dimensions (x, y, z) and not plot_type
            if label_text in DIMENSIONS_TO_PLOT or any(f"{dim} recorded" in label_text for dim in DIMENSIONS_TO_PLOT) or any(f"{dim} calculated" in label_text for dim in DIMENSIONS_TO_PLOT):
                handles.append(handle)
                labels.append(label_text)
            # For entries that are just the dimension names, we can customize the labels
            elif label_text in DIMENSIONS_TO_PLOT:
                handles.append(handle)
                labels.append(f"{label_text} calculated")
        
        # Remove duplicates and create a clean legend
        seen = set()
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        
        plt.legend(unique_handles, unique_labels)
    
    plt.savefig("output/accelerometer_plot.png")
    plt.close()

    print("Accelerometer visualization saved to output/accelerometer_plot.png")
