import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Global variable
SURFACE_TO_USE = "phone"  # Options: "phone" or "hand"

def load_and_prepare_data(file_path):
    """Load ultrasonic data and filter based on surface"""
    print("Loading ultrasonic data...")
    df = pd.read_csv(file_path)
    # Filter by selected surface
    filtered_df = df[df["surface"] == SURFACE_TO_USE]
    return filtered_df

def perform_linear_regression(df):
    """Perform linear regression on distance vs value"""
    X = df[["value"]]
    y = df["distance"]
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    return regressor, r2, y_pred

def create_regression_plot(df, regressor, r2):
    """Create and save regression plot"""
    plt.figure()
    sns.scatterplot(data=df, x="value", y="distance")
    plt.xlabel("Value")
    plt.ylabel("Distance")
    plt.title(f"Ultrasonic Sensor Calibration (Surface: {SURFACE_TO_USE})")
    
    # Prepare data for regression line
    plot_df = df.copy()
    plot_df = plot_df.sort_values("value")
    plot_df["distance_pred"] = regressor.predict(plot_df[["value"]])
    
    # Plot regression line
    sns.lineplot(
        data=plot_df, x="value", y="distance_pred", linestyle="--", linewidth=2
    )
    
    # Add regression metrics to the plot
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    plt.text(
        0.05,
        0.90,
        f"R² = {r2:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", alpha=0.5),
    )
    
    plt.tight_layout()
    plt.savefig(f"output/ultrasonic_regression_surface_{SURFACE_TO_USE}.png", dpi=150)
    plt.close()

def create_lookup_table(df, regressor):
    """Create value->distance lookup table using linear regression"""
    # Get unique values from the data
    unique_values = np.sort(df["value"].unique())
    
    # Predict distance using the regression model
    lookup_data = []
    for value in unique_values:
        distance = regressor.predict([[value]])[0]
        lookup_data.append({
            'value': value,
            'distance': distance
        })
    
    lookup_df = pd.DataFrame(lookup_data)
    return lookup_df

def process():
    """Main function to process ultrasonic data for regression analysis"""
    # Load and prepare data
    df = load_and_prepare_data("data/clean_ultrasonic.csv")
    print(f"Using surface: {SURFACE_TO_USE}")
    print(f"Data points: {len(df)}")
    
    # Perform regression
    regressor, r2, y_pred = perform_linear_regression(df)
    
    print(f"R² score: {r2:.4f}")
    print(f"Regression coefficients: {regressor.coef_[0]:.4f} (slope), {regressor.intercept_:.4f} (intercept)")
    
    # Create plot
    create_regression_plot(df, regressor, r2)
    
    # Create lookup table
    print("Creating lookup table...")
    lookup_df = create_lookup_table(df, regressor)
    filename = f"output/ultrasonic_lookup_table_surface_{SURFACE_TO_USE}.csv"
    lookup_df.to_csv(filename, index=False)
    print(f"Lookup table saved to {filename}")
    print(f"Table shape: {lookup_df.shape}")
