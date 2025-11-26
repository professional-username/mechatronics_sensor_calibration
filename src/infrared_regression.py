import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Global variables
MIN_CUTOFF_DISTANCE = 20  # Distance before which to exclude data
MAX_CUTOFF_DISTANCE = 120  # Distance beyond which to exclude data


def exponential_func(x, a, b, c):
    """Exponential function for curve fitting - decreasing"""
    return a * np.exp(-b * x) + c


def load_and_prepare_data(file_path):
    """Load infrared data"""
    print("Loading infrared data...")
    df = pd.read_csv(file_path)
    return df


def perform_exponential_regression(df):
    """Perform exponential regression on distance vs value"""
    # Sort by distance to ensure proper fitting
    sorted_df = df.sort_values("distance")
    X = sorted_df["distance"].values
    y = sorted_df["value"].values

    # Provide initial parameter guesses to help the fitting process
    # For a decreasing exponential, a should be positive, b positive, c around the minimum value
    initial_guess = [max(y) - min(y), 0.1, min(y)]
    
    # Fit exponential curve
    try:
        popt, pcov = curve_fit(exponential_func, X, y, p0=initial_guess, maxfev=5000)
        a, b, c = popt
        y_pred = exponential_func(X, a, b, c)
        r2 = r2_score(y, y_pred)
        return (a, b, c), r2, y_pred
    except Exception as e:
        print(f"Error in exponential regression: {e}")
        return None, None, None


def create_regression_plot(df, params, r2, dataset):
    """Create and save regression plot"""
    plt.figure()
    # Sort the data for better visualization
    sorted_df = df.sort_values("distance")
    sns.scatterplot(data=sorted_df, x="distance", y="value", alpha=0.5, label="Data points")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.title(f"Infrared Sensor Calibration (Dataset {dataset})")

    # Plot fitted curve
    if params:
        a, b, c = params
        x_plot = np.linspace(sorted_df["distance"].min(), sorted_df["distance"].max(), 300)
        y_plot = exponential_func(x_plot, a, b, c)
        plt.plot(x_plot, y_plot, "r-", linewidth=2, label="Exponential fit")

    # Add regression metrics to the plot
    if params:
        a, b, c = params
        plt.text(
            0.05,
            0.90,
            f"R² = {r2:.4f}\nFunction: {a:.4f} * exp(-{b:.4f} * x) + {c:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", alpha=0.5),
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/infrared_regression_dataset_{dataset}.png", dpi=150)
    plt.close()


def create_lookup_table(df, params, dataset):
    """Create value->distance lookup table"""
    if not params:
        return pd.DataFrame()

    a, b, c = params
    # Get unique values from the data
    unique_values = np.sort(df["value"].unique())

    # For exponential function value = a * np.exp(-b * distance) + c
    # Solve for distance: distance = -ln((value - c) / a) / b
    lookup_data = []
    for value in unique_values:
        # Check if the value is within the valid range for the inverse function
        if (value - c) / a > 0 and not np.isclose((value - c) / a, 0):
            try:
                distance = -np.log((value - c) / a) / b
                # Ensure distance is positive and reasonable
                if distance >= 0 and distance <= 2 * df["distance"].max():
                    lookup_data.append({"value": value, "distance": distance})
            except (ValueError, RuntimeWarning):
                # Skip values that cause issues
                continue
        else:
            # Handle cases where the value is too close to c or invalid
            continue

    lookup_df = pd.DataFrame(lookup_data)
    # Sort by value to make it easier to use
    lookup_df = lookup_df.sort_values("value")
    return lookup_df


def process():
    """Main function to process infrared data for regression analysis"""
    # Load data
    df = load_and_prepare_data("data/clean_infrared.csv")
    
    # Get all unique datasets
    datasets = df["dataset"].unique()
    print(f"Found datasets: {list(datasets)}")
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        # Filter data for current dataset and apply cutoffs
        filtered_df = df[df["dataset"] == dataset]
        filtered_df = filtered_df[
            (filtered_df["distance"] >= MIN_CUTOFF_DISTANCE)
            & (filtered_df["distance"] <= MAX_CUTOFF_DISTANCE)
        ]
        print(
            f"Using distance range: {MIN_CUTOFF_DISTANCE} to {MAX_CUTOFF_DISTANCE}"
        )
        print(f"Data points after filtering: {len(filtered_df)}")

        # Perform regression
        params, r2, y_pred = perform_exponential_regression(filtered_df)

        if params:
            a, b, c = params
            print(f"Exponential parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
            print(f"R² score: {r2:.4f}")

            # Create plot
            create_regression_plot(filtered_df, params, r2, dataset)

            # Create lookup table
            print("Creating lookup table...")
            lookup_df = create_lookup_table(filtered_df, params, dataset)
            filename = f"output/infrared_lookup_table_dataset_{dataset}.csv"
            lookup_df.to_csv(filename, index=False)
            print(f"Lookup table saved to {filename}")
            print(f"Table shape: {lookup_df.shape}")
        else:
            print("Regression analysis failed!")
