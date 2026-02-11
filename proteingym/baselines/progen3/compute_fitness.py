#!/usr/bin/env python3
"""
Script to run DMS protein analysis on multiple CSV files.
Processes all CSV files in a specified directory using the run.sh script.
"""

import argparse
import os
import glob
import subprocess
import sys

from progen3.zero_shot.score_v2 import output_score

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DMS analysis on multiple CSV files.')
    parser.add_argument('--dms_dir', type=str, help='Directory containing CSV files to process')
    parser.add_argument('--model_name', type=str, help='Model name (will be prefixed with Profluent-Bio/)')
    parser.add_argument('--sh_dir', type=str, help='the directory where the run.sh script exists')
    parser.add_argument('--ref_file', type=str, help='the reference file containing the metadata of all DMS assays')
    parser.add_argument('--output_dir', type=str, help='the directory of score output')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if directory exists
    if not os.path.isdir(args.dms_dir):
        print(f"Error: Directory '{args.dms_dir}' does not exist")
        sys.exit(1)
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(args.dms_dir, "*.csv"))
    
    # Check if any CSV files were found
    if not csv_files:
        print(f"No CSV files found in directory '{args.dms_dir}'")
        sys.exit(1)
    
    # Print summary of files to process
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Print progress
        print(f"\nProcessing file {i}/{len(csv_files)}: {filename}")
        print("-" * 50)
        
        # Construct the command
        cmd = ["bash", f"{args.sh_dir}/run_v2.sh", 
               filename, f"Profluent-Bio/{args.model_name}",
               args.dms_dir, args.ref_file, args.output_dir
              ]
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
            # Continue with the next file instead of exiting
            continue

    # Convert the scores to the same format as other PG scores and calculate spearman correlations
    output_score(args.dms_dir, args.output_dir, args.ref_file)

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
