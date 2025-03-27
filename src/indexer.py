import json
import os
import sys
import asyncio
from dotenv import load_dotenv
from utils.utils import (
    add_to_database,
    add_to_score,
    check_existing_username,
    get_tweets_by_username,
)

# Load environment variables
load_dotenv()

# Define how old the file should be before re-fetching (in hours)
DATA_AGE_LIMIT = int(os.getenv("DATA_AGE_LIMIT", 24))


async def main():
    if len(sys.argv) != 6:
        print("❌ Incorrect number of arguments")
        print(
            "ℹ️  Usage: python indexer.py <username> <flag> <contexts> <caller> <transaction>"
        )
        sys.exit(0)

    username = sys.argv[1]
    flag = sys.argv[2]
    ctxs = sys.argv[3].split(",")
    caller = sys.argv[4]
    transaction = sys.argv[5]

    # Check if username has already been indexed
    if check_existing_username(username):
        print(f"ℹ️  User {username} has already been indexed. Skipping fetch")
        sys.exit(0)

    # Fetch tweets
    tweets_data = await get_tweets_by_username(
        username, flag, caller, transaction, ctxs
    )

    # Create directory
    output_dir = f"tweets/{username}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/input.json"

    if not tweets_data:
        # Save empty file
        with open(output_file, "w") as f:
            json.dump([], f, indent=2)

        # Add to score
        add_to_score(username, 0, 0, 0, 0, ctxs)
    else:
        # Save tweets to file
        with open(output_file, "w") as f:
            json.dump(tweets_data, f, indent=2)

        print(f"✅ Tweets saved to {output_file}")

        # Add to database
        for tweet in tweets_data:
            add_to_database(tweet)

        # Add to score
        add_to_score(username, len(tweets_data), 0, 0, 1, ctxs)


if __name__ == "__main__":
    asyncio.run(main())
