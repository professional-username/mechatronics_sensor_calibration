import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Process sensor calibration data and generate visualizations')
    parser.add_argument('--accelerometer', action='store_true', help='Process accelerometer data')
    parser.add_argument('--infrared', action='store_true', help='Process infrared data')
    parser.add_argument('--ultrasonic', action='store_true', help='Process ultrasonic data')
    parser.add_argument('--all', action='store_true', help='Process all data')
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    # Process based on flags
    if args.accelerometer or args.all:
        try:
            from . import accelerometer
            print("Processing accelerometer data...")
            accelerometer.process()
        except ImportError as e:
            print(f"Error importing accelerometer module: {e}")
    
    if args.infrared or args.all:
        try:
            from . import infrared
            print("Processing infrared data...")
            infrared.process()
        except ImportError as e:
            print(f"Error importing infrared module: {e}")
    
    if args.ultrasonic or args.all:
        try:
            from . import ultrasonic
            print("Processing ultrasonic data...")
            ultrasonic.process()
        except ImportError as e:
            print(f"Error importing ultrasonic module: {e}")

if __name__ == "__main__":
    main()
