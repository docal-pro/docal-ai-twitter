import csv
import json
from datetime import datetime
import shutil
import os

def fix_tweet_ids():
    # Load tweet data from tweets/tweets_frank.json
    print("Loading tweets/tweets_frank.json...")
    with open('tweets/tweets_frank.json', 'r') as f:
        # Parse with strict string handling for large integers
        tweets = json.loads(f.read(), parse_int=str)
    
    # Create mapping of createdAt to tweet ID (as string)
    timestamp_to_id = {tweet['createdAt']: str(tweet['id']) for tweet in tweets}
    print(f"Loaded {len(timestamp_to_id)} tweet timestamps")
    
    # Print first few mappings for verification
    first_few = list(timestamp_to_id.items())[:3]
    print("\nFirst few tweet mappings for verification:")
    for ts, tid in first_few:
        print(f"Timestamp: {ts} -> ID: {tid}")
    
    # Create backup of original results/results_frank.csv
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f'backups/results_frank.csv.{backup_time}.bak'
    os.makedirs('backups', exist_ok=True)
    shutil.copy2('results_frank.csv', backup_file)
    print(f"\nCreated backup at {backup_file}")
    
    # Read existing results/results_frank.csv and create new version with string IDs
    print("Processing results/results_frank.csv...")
    rows_processed = 0
    ids_fixed = 0
    
    with open('results/results_frank.csv', 'r') as f_in, \
         open('results/results_with_fixed_ids_frank.csv', 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Read header
        header = next(reader)
        writer.writerow(header)
        
        # Process each row
        for row in reader:
            rows_processed += 1
            created_at = row[1]  # Second column is createdAt
            
            # Get the correct tweet ID from our mapping
            tweet_id = timestamp_to_id.get(created_at, '')
            if tweet_id:
                ids_fixed += 1
            
            # Write row with correct ID from tweets/tweets_frank.json
            writer.writerow([tweet_id] + row[1:])
            
            # Print first few rows for verification
            if rows_processed <= 3:
                print(f"\nRow {rows_processed}:")
                print(f"Timestamp: {created_at}")
                print(f"Original ID: {row[0]}")
                print(f"New ID: {tweet_id}")
            
            if rows_processed % 1000 == 0:
                print(f"Processed {rows_processed} rows, fixed {ids_fixed} IDs")
    
    print(f"\nCompleted processing:")
    print(f"Total rows processed: {rows_processed}")
    print(f"Total IDs fixed: {ids_fixed}")
    
    # Replace original file with new version
    os.replace('results/results_with_fixed_ids_frank.csv', 'results/results_frank.csv')
    print("Updated results/results_frank.csv with fixed tweet IDs")

if __name__ == "__main__":
    fix_tweet_ids() 
