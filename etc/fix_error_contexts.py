import csv
import json
from datetime import datetime
import shutil
import os


def fix_error_contexts(user, function):
    # Load affected tweet IDs
    print(f"Loading tweets/{user}/checkpoints/affected_tweet_ids.txt...")
    with open(f"tweets/{user}/checkpoints/affected_tweet_ids.txt", "r") as f:
        affected_ids = set(line.strip() for line in f)
    print(f"Loaded {len(affected_ids)} affected tweet IDs")

    # Load tweet data from ../tweets/{user}/input.json.json for original text
    print(f"\nLoading tweets/{user}/input.json.json...")
    with open(f"tweets/{user}/input.json.json", "r") as f:
        tweets = json.loads(f.read(), parse_int=str)

    # Create mapping of ID to tweet data
    id_to_tweet = {tweet["id"]: tweet for tweet in tweets}
    print(f"Loaded {len(id_to_tweet)} tweets")

    # Create backup of original results/{user}/{function}.csv
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"results/{user}/backups/{function}.csv.{backup_time}.bak"
    os.makedirs(f"results/{user}/backups", exist_ok=True)
    shutil.copy2(f"results/{user}/{function}.csv", backup_file)
    print(f"\nCreated backup at {backup_file}")

    # Read existing {user}/{function}.csv and create new version with fixed error messages
    print(f"Processing results/{user}/{function}.csv...")
    rows_processed = 0
    errors_fixed = 0

    with open(f"results/{user}/{function}.csv", "r") as f_in, open(
        f"results/{user}/{function}_with_fixed_errors.csv", "w", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read header
        header = next(reader)
        writer.writerow(header)

        # Process each row
        for row in reader:
            rows_processed += 1
            tweet_id = row[0]
            context = row[4]

            # Check if this is an affected tweet with an error
            if tweet_id in affected_ids and any(
                error in context.lower()
                for error in [
                    "restricted",
                    "deleted",
                    "suspended",
                    "not accessible",
                    "failed to fetch",
                ]
            ):
                # Get original tweet data
                tweet_data = id_to_tweet.get(tweet_id)
                if tweet_data:
                    # Create new context with original text
                    new_context = (
                        f"[This post is unavailable] @{user}: {tweet_data['text']}"
                    )
                    row[4] = new_context
                    errors_fixed += 1

            # Write row (either original or fixed)
            writer.writerow(row)

            # Print first few rows for verification
            if rows_processed <= 3:
                print(f"\nRow {rows_processed}:")
                print(f"Tweet ID: {tweet_id}")
                print(f"Original context: {context}")
                print(f"New context: {row[4]}")

            if rows_processed % 1000 == 0:
                print(f"Processed {rows_processed} rows, fixed {errors_fixed} errors")

    print(f"\nCompleted processing:")
    print(f"Total rows processed: {rows_processed}")
    print(f"Total errors fixed: {errors_fixed}")

    # Replace original file with new version
    os.replace(
        f"results/{user}/{function}_with_fixed_errors.csv",
        f"results/{user}/{function}.csv",
    )
    print(f"Updated results/{user}/{function}.csv with fixed error messages")


if __name__ == "__main__":
    fix_error_contexts()
