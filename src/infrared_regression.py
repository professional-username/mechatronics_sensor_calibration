import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Global variables
DATASET_TO_USE = "A"  # Options: "A" or "B"
CUTOFF_DISTANCE = 120  # Distance beyond which to exclude data

def exponential_func(x, a, b, c):
    """Exponential function for curve fitting"""
    return a * np.exp(b * x) + c

def load_and_prepare_data(file_path):
    """Load infrared data and filter based on dataset and cutoff"""
    print("Loading infrared data...")
    df = pd.read_csv(file_path)
    # Filter by selected dataset
    filtered_df = df[df["dataset"] == DATASET_TO_USE]
    # Apply distance cutoff
    filtered_df = filtered_df[filtered_df["distance"] <= CUTOFF_DISTANCE]
    return filtered_df

def perform_exponential_regression(df):
    """Perform exponential regression on distance vs value"""
    X = df["distance"].values
    y = df["value"].values
    
    # Fit exponential curve
    try:
        popt, pcov = curve_fit(exponential_func, X, y, maxfev=5000)
        a, b, c = popt
        y_pred = exponential_func(X, a, b, c)
        r2 = r2_score(y, y_pred)
        return (a, b, c), r2, y_pred
    except Exception as e:
        print(f"Error in exponential regression: {e}")
        return None, None, None

def create_regression_plot(df, params, r2):
    """Create and save regression plot"""
    plt.figure()
    sns.scatterplot(data=df, x="distance", y="value")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.title(f"Infrared Sensor Calibration (Dataset {DATASET_TO_USE})")
    
    # Plot fitted curve
    if params:
        a, b, c = params
        x_plot = np.linspace(df["distance"].min(), df["distance"].max(), 100)
        y_plot = exponential_func(x_plot, a, b, c)
        plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Exponential fit')
    
    # Add regression metrics to the plot
    if params:
        a, b, c = params
        plt.text(
            0.05,
            0.90,
            f"R² = {r2:.4f}\nFunction: {a:.4f} * exp({b:.4f} * x) + {c:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", alpha=0.5),
        )
    
    plt.tight_layout()
    plt.savefig(f"output/infrared_regression_dataset_{DATASET_TO_USE}.png", dpi=150)
    plt.close()

def create_lookup_table(df, params):
    """Create value->distance lookup table"""
    if not params:
        return pd.DataFrame()
    
    a, b, c = params
    # Get unique values from the data
    unique_values = np.sort(df["value"].unique())
    
    # For exponential function value = a * exp(b * distance) + c
    # Solve for distance: distance = ln((value - c) / a) / b
    lookup_data = []
    for value in unique_values:
        if (value - c) / a > 0:  # Ensure valid for logarithm
            distance = np.log((value - c) / a) / b
            lookup_data.append({
                'value': value,
                'distance': distance
            })
    
    lookup_df = pd.DataFrame(lookup_data)
    return lookup_df

def process():
    """Main function to process infrared data for regression analysis"""
    # Load and prepare data
    df = load_and_prepare_data("data/clean_infrared.csv")
    print(f"Using dataset {DATASET_TO_USE} with distance cutoff {CUTOFF_DISTANCE}")
    print(f"Data points after filtering: {len(df)}")
    
    # Perform regression
    params, r2, y_pred = perform_exponential_regression(df)
    
    if params:
        a, b, c = params
        print(f"Exponential parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        print(f"R² score: {r2:.4f}")
        
        # Create plot
        create_regression_plot(df, params, r2)
        
        # Create lookup table
        print("Creating lookup table...")
        lookup_df = create_lookup_table(df, params)
        filename = f"output/infrared_lookup_table_dataset_{DATASET_TO_USE}.csv"
        lookup_df.to_csv(filename, index=False)
        print(f"Lookup table saved to {filename}")
        print(f"Table shape: {lookup_df.shape}")
    else:
        print("Regression analysis failed!")
