import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def process():
    # Load data
    df = pd.read_csv("data/calibrated_ultrasonic.csv")
    print(f"Loaded calibrated ultrasonic data with shape: {df.shape}")

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot measured data as scatter
    sns.scatterplot(data=df, x="distance", y=" measured distance", alpha=0.7)
    
    # Plot perfect line (y=x) as dotted line
    min_val = df["distance"].min()
    max_val = df["distance"].max()
    perfect_x = np.linspace(min_val, max_val, 100)
    perfect_y = perfect_x
    plt.plot(perfect_x, perfect_y, linestyle=':', linewidth=2, label='Perfect calibration (y=x)')
    
    plt.title("Ultrasonic Sensor Calibration Accuracy")
    plt.xlabel("Actual Distance")
    plt.ylabel("Measured Distance")
    plt.legend()
    plt.savefig("output/plots/ultrasonic/ultrasonic_calibrated_plot.png")
    plt.close()
    
    print("Ultrasonic calibration plot saved to output/plots/ultrasonic/ultrasonic_calibrated_plot.png")
