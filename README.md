# Sensor Data Processing and Calibration

This project processes and calibrates sensor data from accelerometer, infrared, and ultrasonic sensors. It generates visualizations and lookup tables for sensor calibration.

## Prerequisites

- Nix package manager (for development environment)
- Raw sensor data files (not included in the repository)

## Setup

1. Install the development environment:
   ```bash
   devenv shell
   ```

2. Install Python dependencies:
   ```bash
   uv sync
   ```

## Data Requirements

Place your raw data files in the `data/` directory. The following files are expected:
- `clean_accelerometer.csv`
- `clean_infrared.csv`
- `clean_ultrasonic.csv`
- `calibrated_accelerometer.csv`
- `calibrated_infrared.csv`

Note: The raw data files are not provided with this project and must be added separately.

## Usage

Run the main processing script with various flags to process different sensor data:

```bash
python -m src.main [OPTIONS]
```

### Available Options

- `--accelerometer`: Process accelerometer data
- `--infrared`: Process infrared data
- `--ultrasonic`: Process ultrasonic data
- `--accelerometer-regression`: Process accelerometer regression analysis
- `--infrared-regression`: Process infrared regression analysis
- `--ultrasonic-regression`: Process ultrasonic regression analysis
- `--accelerometer-calibrated`: Plot calibrated accelerometer data
- `--infrared-calibrated`: Plot calibrated infrared data
- `--ultrasonic-calibrated`: Plot calibrated ultrasonic data
- `--all`: Process all data types

### Examples

Process all data:
```bash
python -m src.main --all
```

Process only accelerometer data:
```bash
python -m src.main --accelerometer
```

Process regression analyses:
```bash
python -m src.main --accelerometer-regression --infrared-regression --ultrasonic-regression
```

## Output

The processed data and visualizations are saved in the `output/` directory:
- `output/plots/`: Contains generated plots organized by sensor type
- `output/lookup_tables/`: Contains calibration lookup tables

## Development

The project uses:
- Python with uv for dependency management
- devenv for development environment management
- pandas for data processing
- seaborn and matplotlib for visualization
- scikit-learn for regression analysis
