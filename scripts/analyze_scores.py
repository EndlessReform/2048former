#!/usr/bin/env python3

import os
import sys
import sqlite3
import argparse
from collections import defaultdict

def get_scores_from_db(db_path):
    """Extract scores from a SQLite database file."""
    try:
        # Create a copy of the database to avoid permission issues
        temp_db = '/tmp/metadata_copy.db'
        if os.path.exists(temp_db):
            os.remove(temp_db)
        
        # Copy the db file
        with open(db_path, 'rb') as src, open(temp_db, 'wb') as dst:
            dst.write(src.read())
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Query all max scores
        cursor.execute("SELECT max_score FROM runs")
        scores = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        os.remove(temp_db)  # Clean up
        
        return scores
    except Exception as e:
        print(f"Error reading database {db_path}: {e}")
        return []

def calculate_quantiles(scores, quantiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
    """Calculate specified quantiles for a list of scores."""
    if not scores:
        return {}
    
    # Sort scores
    sorted_scores = sorted(scores)
    
    result = {}
    for q in quantiles:
        idx = int(q * len(sorted_scores))
        if idx >= len(sorted_scores):
            idx = len(sorted_scores) - 1
        result[f"{q*100:.0f}%"] = sorted_scores[idx]
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Analyze game scores from SQLite database files')
    parser.add_argument('folder', help='Path to folder containing metadata.db files')
    parser.add_argument('--output', '-o', help='Output file for results (optional)', default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist")
        sys.exit(1)
    
    # Find all metadata.db files in the folder
    db_files = []
    for root, dirs, files in os.walk(args.folder):
        for file in files:
            if file == 'metadata.db':
                db_files.append(os.path.join(root, file))
    
    if not db_files:
        print(f"No metadata.db files found in {args.folder}")
        sys.exit(1)
    
    all_scores = []
    
    for db_file in db_files:
        print(f"Processing {db_file}...")
        scores = get_scores_from_db(db_file)
        if scores:
            all_scores.extend(scores)
            print(f"  Found {len(scores)} scores")
        else:
            print(f"  No scores found in {db_file}")
    
    if not all_scores:
        print("No scores to analyze")
        sys.exit(1)
    
    print(f"\nTotal scores processed: {len(all_scores)}")
    
    # Calculate statistics
    min_score = min(all_scores)
    max_score = max(all_scores)
    avg_score = sum(all_scores) / len(all_scores)
    
    quantiles = calculate_quantiles(all_scores)
    
    # Print results
    print(f"\n--- Game Score Statistics ---")
    print(f"Minimum score: {min_score:,}")
    print(f"Maximum score: {max_score:,}")
    print(f"Average score: {avg_score:,.2f}")
    
    print(f"\nQuantiles:")
    for q, score in quantiles.items():
        print(f"  {q}: {score:,}")
    
    # Write to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write("--- Game Score Statistics ---\n")
            f.write(f"Minimum score: {min_score:,}\n")
            f.write(f"Maximum score: {max_score:,}\n")
            f.write(f"Average score: {avg_score:,.2f}\n")
            
            f.write("\nQuantiles:\n")
            for q, score in quantiles.items():
                f.write(f"  {q}: {score:,}\n")
        print(f"\nResults written to {args.output}")
    
if __name__ == '__main__':
    main()
