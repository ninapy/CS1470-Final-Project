import numpy as np
import pandas as pd
import os
import sys
import time
import random

# Preprocessing data, reducing 4.5 million to 200k from WMT EN-DE dataset

TRAIN_FILE = './data/wmt14_translate_de-en_train.csv'
OUTPUT_FILE = './data/wmt14_reduced_200k.csv'
TARGET_SIZE = 200000

#Using Reservoir Random Sampling because of the size of our data n=4.5million
#and we want a representative 200k sample.

# Set random seed for reproducibility
# ensures the sampling results will be the same every time we run the script
random.seed(42)

def reservoir_sampling(input_file, output_file, target_size):
    """
    Implement reservoir sampling algorithm for very large files.
    This is a one-pass algorithm that doesn't require knowing file size before.

    Reservoir sampling works by:
    1. Keeping the first 'target_size' items
    2. For each subsequent item i:
       - Generate random number j between 0 and i
       - If j < target_size, replace item j in reservoir with current item

    This ensures each item has equal probability (target_size/total) of being selected.

    Args:
        input_file (str): Path to input CSV train file from WMT2014 en-de
        output_file (str): Path to save the reduced dataset
        target_size (int): Number of rows to sample
    """
    print(f'Start of sampling from {input_file}')
    start_time = time.time()

    reservoir = []

    with open(input_file, 'rb') as f:
        # Read header
        header = f.readline().decode('utf-8', errors='replace').strip()
        
        # Fill reservoir with first target_size items
        for i in range(target_size):
            line = f.readline()
            if not line:  # End of file
                break
            reservoir.append(line.decode('utf-8', errors='replace').strip())
        
        print(f"Initial reservoir filled with {len(reservoir)} items")
        
        # Process remaining items with decreasing probability
        i = target_size
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            
            i += 1
            if i % 1000000 == 0:
                print(f"Processed {i:,} lines...")
            
            # With probability target_size/i, replace a random item in the reservoir
            j = random.randrange(i)
            if j < target_size:
                reservoir[j] = line.decode('utf-8', errors='replace').strip()
    
    # Write reservoir to output file
    print(f"Writing {len(reservoir)} lines to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for line in reservoir:
            f.write(line + '\n')
    
    elapsed = time.time() - start_time
    print(f"Reservoir sampling completed in {elapsed:.2f} seconds")
    print(f"Reduced dataset saved to {output_file}")

def check_output_file(file_path, expected_count=None):
    """Verify the output file and print some statistics."""
    print(f"\nVerifying output file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("ERROR: Output file does not exist!")
        return
    
    # Check file size
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    # Count lines
    line_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in f:
            line_count += 1
    
    print(f"Total lines (including header): {line_count}")
    
    if expected_count and line_count != expected_count + 1:  # +1 for header
        print(f"WARNING: Expected {expected_count + 1} lines but found {line_count}")
    
    # Print a few sample lines
    print("\nSample content:")
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline().strip()
        print(f"Header: {header}")
        
        print("\nFirst 3 data rows:")
        for i in range(3):
            line = f.readline().strip()
            print(f"{i+1}: {line}")
    
    print("\nOutput file verification complete.")

def main():
    """Main function to execute sampling with error handling."""
    print(f"Starting dataset reduction to {TARGET_SIZE:,} rows...")
    
    try:
        reservoir_sampling(TRAIN_FILE, OUTPUT_FILE, TARGET_SIZE)
        # Verify the output file
        check_output_file(OUTPUT_FILE, TARGET_SIZE)
        print("\nDataset reduction completed successfully!")
    except Exception as e:
        print(f"Error during dataset reduction: {e}")

if __name__ == "__main__":
    main()
