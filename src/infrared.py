import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process():
    # Load data
    df = pd.read_csv("data/clean_infrared.csv")
    print(f"Loaded infrared data with shape: {df.shape}")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot by dataset
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        sns.lineplot(data=dataset_data, x='distance', y='value', label=f'Dataset {dataset}')
    
    plt.title("Infrared Sensor Data by Distance and Dataset")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/infrared_plot.png")
    plt.close()
    
    print("Infrared visualization saved to output/infrared_plot.png")
