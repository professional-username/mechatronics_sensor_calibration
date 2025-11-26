import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process():
    # Load data
    df = pd.read_csv("data/clean_ultrasonic.csv")
    print(f"Loaded ultrasonic data with shape: {df.shape}")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot by surface type
    for surface in df['surface'].unique():
        surface_data = df[df['surface'] == surface]
        sns.lineplot(data=surface_data, x='distance', y='value', label=f'Surface: {surface}')
    
    plt.title("Ultrasonic Sensor Data by Distance and Surface")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/plots/ultrasonic/ultrasonic_plot.png")
    plt.close()
    
    print("Ultrasonic visualization saved to output/plots/ultrasonic/ultrasonic_plot.png")
