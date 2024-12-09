import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import math

CONTEXT_WINDOW = 64
CHUNK_SIZE = 72  # Number of CSV files to process in each chunk
N_LAST_CHUNKS = 1  # Number of chunks to use for validation and test sets

def zigzag_permute(sequence):
    """
    Permutes a sequence of coordinates in a zigzag pattern.
    
    Args:
        sequence: numpy array of shape (CONTEXT_WINDOW, 2) with possible NaN values at the end
        
    Returns:
        numpy array of shape (CONTEXT_WINDOW, 2) with zigzag permutation of non-NaN values
    """
    # Find indices of non-NaN rows
    valid_mask = ~np.isnan(sequence).any(axis=1)
    valid_coords = sequence[valid_mask]
    n_valid = len(valid_coords)
    
    # Create empty result array filled with NaNs
    result = np.full((CONTEXT_WINDOW, 2), np.nan)
    
    if n_valid == 0:
        return result
    
    # Create zigzag indices
    zigzag_idx = []
    left, right = 0, n_valid - 1
    
    while left <= right:
        zigzag_idx.append(left)
        if left != right:
            zigzag_idx.append(right)
        left += 1
        right -= 1
    
    # Fill result with permuted coordinates
    result[:n_valid] = valid_coords[zigzag_idx]
    
    return result

def process_csv_chunk(csv_files, data_dir, zigzag=True):
    """
    Process a chunk of CSV files and return the processed sequences.
    
    Args:
        csv_files: List of CSV filenames to process
        data_dir: Directory containing the CSV files
    
    Returns:
        List of processed sequences
    """
    chunk_data = []
    for filename in tqdm(csv_files):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        for unique_id in df["id"].unique():
            subset = df[df["id"] == unique_id]
            sequence = subset[["from_lat", "from_lon"]].values  # (n, 2)
            final_row = subset[["to_lat", "to_lon"]].iloc[-1].values  # (2,)

            # Add the final row as a new entry
            sequence = np.concatenate((sequence, final_row.reshape(1, 2)))

            # Pad the sequence with NaNs if it's shorter than CONTEXT_WINDOW
            padding_length = CONTEXT_WINDOW - len(sequence)
            if padding_length > 0:
                padding = np.full((padding_length, 2), np.nan)
                sequence = np.concatenate((sequence, padding))
            elif padding_length < 0:
                sequence = sequence[:CONTEXT_WINDOW]

            if zigzag:
                chunk_data.append(zigzag_permute(sequence))
            else:
                chunk_data.append(sequence)
    
    return chunk_data

def build_datasets(data_dir="data/csv", output_dir="data/processed",
                   val_size=0.1, test_size=0.1, random_state=42, zigzag=True):
    """
    Builds training, validation, and test datasets from CSV files in chunks.

    Args:
        data_dir: Directory containing the CSV files.
        output_dir: Directory to save the processed datasets.
        val_size: Proportion of the dataset to include in the validation split.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.
        zigzag: Whether to apply zigzag permutation to the sequences.

    Returns:
        The number of train chunks created.
    """
    # Get list of CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("._")]

    # Sort the CSV files by their names
    csv_files.sort()
    
    # Calculate number of chunks
    num_chunks = math.ceil(len(csv_files) / CHUNK_SIZE)

    print(f'Number of chunks: {num_chunks}')
    
    # Process the last N chunks separately to create val and test sets
    last_chunks_start = max(0, (num_chunks - N_LAST_CHUNKS) * CHUNK_SIZE)
    main_chunks_files = csv_files[:last_chunks_start]
    last_chunks_files = csv_files[last_chunks_start:]
    
    print(f'Number of main chunks: {len(main_chunks_files)}')
    print(f'Number of last chunks: {len(last_chunks_files)}')
    
    # Process main chunks and save training sets
    for chunk_idx in range(0, len(main_chunks_files), CHUNK_SIZE):
        chunk_files = main_chunks_files[chunk_idx:chunk_idx + CHUNK_SIZE]
        if not chunk_files:  # Skip if no files in chunk
            continue
            
        chunk_data = process_csv_chunk(chunk_files, data_dir, zigzag)
        if chunk_data:  # Only save if we have data
            chunk_data = np.array(chunk_data)
            chunk_number = (chunk_idx // CHUNK_SIZE) + 1
            save_train_chunk(chunk_data, chunk_number, output_dir)
    
    # Process last chunks for validation and test sets
    last_chunks_data = process_csv_chunk(last_chunks_files, data_dir, zigzag)
    if last_chunks_data:
        last_chunks_data = np.array(last_chunks_data)
        
        # Split the last chunks into validation and test sets
        val_data, test_data = train_test_split(
            last_chunks_data, 
            test_size=test_size/(test_size + val_size), 
            random_state=random_state
        )
        
        # Save validation and test sets
        save_validation_and_testdatasets(val_data, test_data, output_dir)
        
        return (len(main_chunks_files) // CHUNK_SIZE)
    
    return (len(main_chunks_files) // CHUNK_SIZE)

def save_train_chunk(data, chunk_number, output_dir):
    """
    Saves a training data chunk to a numbered file.

    Args:
        data: Training data chunk
        chunk_number: Chunk number for filename
        output_dir: Directory to save the dataset to
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"train.{chunk_number:02d}.ds.pt"
    torch.save(data, os.path.join(output_dir, filename))


def save_validation_and_testdatasets(val_data, test_data, output_dir="data/processed"):
    """
    Saves the validation and test datasets to files.

    Args:
        val_data: Validation dataset
        test_data: Test dataset
        output_dir: Directory to save the datasets to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the validation and test datasets
    torch.save(val_data, os.path.join(output_dir, "val.ds.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.ds.pt"))

