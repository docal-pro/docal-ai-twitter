import csv
import json
from datetime import datetime
import shutil
import os


def add_tweet_ids(user, function):
    # Load tweet data from tweets/{user}/input.json
    print("Loading tweets/{user}/input.json...")
    with open("../tweets/{user}/input.json", "r") as f:
        # Parse with strict string handling for large integers
        tweets = json.loads(f.read(), parse_int=str)

    # Create mapping of ID to tweet data
    id_to_tweet = {tweet["id"]: tweet for tweet in tweets}
    print(f"Loaded {len(id_to_tweet)} tweets")

    # Print first few tweets for verification
    first_few = list(id_to_tweet.items())[:3]
    print("\nFirst few tweets for verification:")
    for tid, tweet in first_few:
        print(f"ID: {tid}")
        print(f"CreatedAt: {tweet['createdAt']}")
        print(f"Text: {tweet['text'][:50]}...")
        print()

    # Create backup of original results/{user}/{function}.csv
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"../backups/{user}/{function}.csv.{backup_time}.bak"
    os.makedirs("../backups", exist_ok=True)
    shutil.copy2("../results/{user}/{function}.csv", backup_file)
    print(f"\nCreated backup at {backup_file}")

    # Read existing results/{user}/{function}.csv and create new version with corrected columns
    print("Processing results/{user}/{function}.csv...")
    rows_processed = 0
    matches_found = 0

    with open("../results/{user}/{function}.csv", "r") as f_in, open(
        "../results/{user}/{function}_with_ids.csv", "w", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Skip the header row with duplicate tweet_id columns
        header = next(reader)
        # Write new header with single tweet_id column
        writer.writerow(["tweet_id", "createdAt", "text", "isReply", "context"])

        # Process each row
        for row in reader:
            rows_processed += 1
            tweet_id = row[2]  # Tweet ID is in the third column

            # Verify the tweet exists in our mapping
            if tweet_id in id_to_tweet:
                matches_found += 1
                # Write row with correct columns
                writer.writerow([tweet_id, row[3], row[4], row[5], row[6]])
            else:
                # If no match, write empty ID but keep other data
                writer.writerow(["", row[3], row[4], row[5], row[6]])

            # Print first few rows for verification
            if rows_processed <= 3:
                print(f"\nRow {rows_processed}:")
                print(f"Tweet ID: {tweet_id}")
                print(f"CreatedAt: {row[3]}")
                print(f"Text: {row[4]}")

            if rows_processed % 1000 == 0:
                print(f"Processed {rows_processed} rows, found {matches_found} matches")

    print(f"\nCompleted processing:")
    print(f"Total rows processed: {rows_processed}")
    print(f"Total matches found: {matches_found}")

    # Replace original file with new version
    os.replace(
        "../results/{user}/{function}_with_ids.csv", "../results/{user}/{function}.csv"
    )
    print("Updated {user}/{function}.csv with corrected columns")


if __name__ == "__main__":
    add_tweet_ids()
