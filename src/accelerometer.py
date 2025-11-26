import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process():
    # Load data
    df = pd.read_csv("data/clean_acceletometer.csv")
    print(f"Loaded accelerometer data with shape: {df.shape}")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Separate recorded and calculated data
    recorded = df[df['data type'] == 'recorded']
    calculated = df[df['data type'] == 'calculated']
    
    # Plot recorded data by dimension
    for dim in ['x', 'y', 'z']:
        dim_data = recorded[recorded['dimention'] == dim]
        sns.lineplot(data=dim_data, x='angle', y='value', label=f'{dim} recorded')
    
    # Plot calculated data by dimension
    for dim in ['x', 'y', 'z']:
        dim_data = calculated[calculated['dimention'] == dim]
        sns.lineplot(data=dim_data, x='angle', y='value', label=f'{dim} calculated', linestyle='--')
    
    plt.title("Accelerometer Data by Angle and Dimension")
    plt.xlabel("Angle")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/accelerometer_plot.png")
    plt.close()
    
    print("Accelerometer visualization saved to output/accelerometer_plot.png")
