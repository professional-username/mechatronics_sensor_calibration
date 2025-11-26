import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_csv_data(file_path):
    """Load and return CSV data as pandas DataFrame"""
    return pd.read_csv(file_path)

def create_visualizations(df, output_dir="output"):
    """Create various seaborn plots from the DataFrame"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Example plots - customize based on your data
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df)
    plt.title("Sensor Data Over Time")
    plt.savefig(f"{output_dir}/lineplot.png")
    plt.close()
    
    # Add more plot types as needed
    # sns.scatterplot, sns.histplot, sns.heatmap, etc.

def main():
    print("Processing sensor calibration data...")
    
    # Example usage - modify paths as needed
    data_file = "data/sensor_data.csv"
    
    try:
        df = load_csv_data(data_file)
        print(f"Loaded data with shape: {df.shape}")
        
        create_visualizations(df)
        print("Visualizations saved to output/ directory")
        
    except FileNotFoundError:
        print(f"Data file {data_file} not found")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
