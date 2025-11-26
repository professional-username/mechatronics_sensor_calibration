import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_and_prepare_data(file_path):
    """Load accelerometer data and separate recorded vs calculated values"""
    print("Loading accelerometer data...")
    df = pd.read_csv(file_path)
    recorded_df = df[df["data type"] == "recorded"]
    calculated_df = df[df["data type"] == "calculated"]
    return recorded_df, calculated_df


def prepare_dimension_data(recorded_df, calculated_df, dimension):
    """Prepare data for a specific dimension by merging recorded and calculated values"""
    # Filter data for current dimension
    recorded_dim = recorded_df[recorded_df["dimension"] == dimension]
    calculated_dim = calculated_df[calculated_df["dimension"] == dimension]
    
    # Group by angle and take mean values
    recorded_agg = recorded_dim.groupby("angle")["value"].mean().reset_index()
    calculated_agg = calculated_dim.groupby("angle")["value"].mean().reset_index()
    
    # Merge on angle
    merged_df = pd.merge(
        recorded_agg,
        calculated_agg,
        on="angle",
        suffixes=("_recorded", "_calculated"),
    )
    return merged_df


def perform_regression_analysis(merged_df):
    """Perform linear regression and return model and metrics"""
    X = merged_df[["value_recorded"]]
    y = merged_df["value_calculated"]
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    return regressor, r2, y_pred


def create_regression_plot(merged_df, dimension, regressor, r2):
    """Create and save regression plot for a dimension"""
    # Create scatter plot
    plt.figure()
    sns.scatterplot(data=merged_df, x="value_recorded", y="value_calculated")
    plt.xlabel(f"Recorded {dimension} values")
    plt.ylabel(f"Calculated {dimension} values")
    plt.title(f"Recorded vs Calculated {dimension.upper()} Values")
    
    # Prepare data for regression line
    plot_df = merged_df.copy()
    plot_df = plot_df.sort_values("value_recorded")
    plot_df["y_pred"] = regressor.predict(plot_df[["value_recorded"]])
    
    # Plot regression line
    sns.lineplot(
        data=plot_df, x="value_recorded", y="y_pred", linestyle="--", linewidth=2
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
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"output/accelerometer_regression_{dimension}.png", dpi=150)
    plt.close()


def process_dimension(recorded_df, calculated_df, dimension):
    """Process a single dimension and return regression metrics"""
    print(f"Processing {dimension} dimension...")
    
    # Prepare data
    merged_df = prepare_dimension_data(recorded_df, calculated_df, dimension)
    
    # Perform regression
    regressor, r2, y_pred = perform_regression_analysis(merged_df)
    
    # Create plot
    create_regression_plot(merged_df, dimension, regressor, r2)
    
    # Print metrics
    print(f"  R² score for {dimension}: {r2:.4f}")
    print(
        f"  Regression coefficients: {regressor.coef_[0]:.4f} (slope), {regressor.intercept_:.4f} (intercept)"
    )
    
    return {
        'dimension': dimension,
        'r2': r2,
        'slope': regressor.coef_[0],
        'intercept': regressor.intercept_
    }


def create_lookup_table(recorded_df, calculated_df, results):
    """Create a lookup table with angles, calculated values, and measured values using regression"""
    # Get all unique angles
    all_angles = pd.concat([
        recorded_df['angle'], 
        calculated_df['angle']
    ]).unique()
    all_angles.sort()
    
    lookup_data = []
    
    for angle in all_angles:
        row = {'angle': angle}
        
        # For each dimension, get the calculated value and compute measured value
        for result in results:
            dimension = result['dimension']
            slope = result['slope']
            intercept = result['intercept']
            
            # Get calculated value for this angle and dimension
            calc_mask = (calculated_df['angle'] == angle) & (calculated_df['dimension'] == dimension)
            if calc_mask.any():
                calc_value = calculated_df.loc[calc_mask, 'value'].iloc[0]
            else:
                calc_value = None
            
            # Compute measured value using regression: measured = (calculated - intercept) / slope
            # But be careful if slope is zero
            if calc_value is not None and slope != 0:
                measured_value = (calc_value - intercept) / slope
            else:
                measured_value = None
            
            row[f'{dimension}_calculated'] = calc_value
            row[f'{dimension}_measured'] = measured_value
        
        lookup_data.append(row)
    
    lookup_df = pd.DataFrame(lookup_data)
    return lookup_df


def process():
    """Main function to process accelerometer data for regression analysis"""
    # Load data
    recorded_df, calculated_df = load_and_prepare_data("data/clean_accelerometer.csv")
    
    # Process each dimension
    dimensions = ["x", "y", "z"]
    results = []
    
    for dimension in dimensions:
        result = process_dimension(recorded_df, calculated_df, dimension)
        results.append(result)
    
    # Create and save lookup table
    print("\nCreating lookup table...")
    lookup_df = create_lookup_table(recorded_df, calculated_df, results)
    lookup_df.to_csv("output/accelerometer_lookup_table.csv", index=False)
    print(f"Lookup table saved to output/accelerometer_lookup_table.csv")
    print(f"Table shape: {lookup_df.shape}")
    
    # Print summary
    print("\nRegression analysis complete!")
    print("Summary of results:")
    for result in results:
        print(f"  {result['dimension'].upper()}: R²={result['r2']:.4f}, "
              f"slope={result['slope']:.4f}, intercept={result['intercept']:.4f}")
