import argparse
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


def configure_seaborn():
    """Configure seaborn style settings"""
    sns.set_theme()
    # sns.set_palette("mako")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["figure.dpi"] = 100


def setup_environment():
    """Create output directory and configure plotting"""
    Path("output").mkdir(exist_ok=True)
    configure_seaborn()


def parse_arguments():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process sensor calibration data and generate visualizations"
    )
    parser.add_argument(
        "--accelerometer", action="store_true", help="Process accelerometer data"
    )
    parser.add_argument("--infrared", action="store_true", help="Process infrared data")
    parser.add_argument(
        "--ultrasonic", action="store_true", help="Process ultrasonic data"
    )
    parser.add_argument(
        "--accelerometer-regression", action="store_true", 
        help="Process accelerometer regression analysis"
    )
    parser.add_argument(
        "--infrared-regression", action="store_true", 
        help="Process infrared regression analysis"
    )
    parser.add_argument(
        "--ultrasonic-regression", action="store_true", 
        help="Process ultrasonic regression analysis"
    )
    parser.add_argument("--all", action="store_true", help="Process all data")
    return parser.parse_args()


def process_sensor_modules(args):
    """Process the appropriate sensor modules based on arguments"""
    should_process_sensor = lambda a, s: getattr(a, s) or a.all
    if should_process_sensor(args, "accelerometer"):
        try:
            from . import accelerometer

            print("Processing accelerometer data...")
            accelerometer.process()
        except ImportError as e:
            print(f"Error importing accelerometer module: {e}")

    if should_process_sensor(args, "infrared"):
        try:
            from . import infrared

            print("Processing infrared data...")
            infrared.process()
        except ImportError as e:
            print(f"Error importing infrared module: {e}")

    if should_process_sensor(args, "ultrasonic"):
        try:
            from . import ultrasonic

            print("Processing ultrasonic data...")
            ultrasonic.process()
        except ImportError as e:
            print(f"Error importing ultrasonic module: {e}")

    if should_process_sensor(args, "accelerometer_regression"):
        try:
            from . import accelerometer_regression

            print("Processing accelerometer regression analysis...")
            accelerometer_regression.process()
        except ImportError as e:
            print(f"Error importing accelerometer regression module: {e}")

    if should_process_sensor(args, "infrared_regression"):
        try:
            from . import infrared_regression

            print("Processing infrared regression analysis...")
            infrared_regression.process()
        except ImportError as e:
            print(f"Error importing infrared regression module: {e}")

    if should_process_sensor(args, "ultrasonic_regression"):
        try:
            from . import ultrasonic_regression

            print("Processing ultrasonic regression analysis...")
            ultrasonic_regression.process()
        except ImportError as e:
            print(f"Error importing ultrasonic regression module: {e}")


def main():
    """Main entry point for the CLI"""
    args = parse_arguments()

    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parse_arguments().print_help()
        return

    setup_environment()
    process_sensor_modules(args)


if __name__ == "__main__":
    main()
