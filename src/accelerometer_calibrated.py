import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def process():
    # Load data
    df = pd.read_csv("data/calibrated_acceletometer.csv")
    print(f"Loaded calibrated accelerometer data with shape: {df.shape}")

    # Separate recorded and calculated data
    recorded = df[df["data type"] == "recorded"]
    calculated = df[df["data type"] == "calculated"]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot recorded data as scatter
    sns.scatterplot(data=recorded, x="angle", y="reading", hue="dimension", alpha=0.7)
    
    # Plot calculated data as dotted line (perfect calibration)
    # Sort by angle to ensure proper line plotting
    calculated_sorted = calculated.sort_values('angle')
    for dimension in calculated_sorted['dimension'].unique():
        dimension_data = calculated_sorted[calculated_sorted['dimension'] == dimension]
        plt.plot(dimension_data['angle'], dimension_data['reading'], 
                linestyle=':', linewidth=2, label=f'{dimension} (perfect)')
    
    plt.title("Accelerometer Calibration Accuracy")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reading")
    plt.legend()
    plt.savefig("output/plots/accelerometer/accelerometer_calibrated_plot.png")
    plt.close()
    
    print("Accelerometer calibration plot saved to output/plots/accelerometer/accelerometer_calibrated_plot.png")
