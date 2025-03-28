import json
import os
import sys
import asyncio
from dotenv import load_dotenv
from utils.utils import (
    add_to_database,
    add_to_score,
    check_existing_tweets,
    get_tweets_by_ids,
)

# Load environment variables
load_dotenv()


async def main():
    if len(sys.argv) != 7:
        print("❌ Incorrect number of arguments")
        print(
            "ℹ️  Usage: python scraper.py <username> <tweet_ids> <flag> <contexts> <caller> <transaction>"
        )
        sys.exit(0)

    # Parse comma-separated tweet IDs
    username = sys.argv[1]
    tweet_ids = sys.argv[2].split(",")
    flag = sys.argv[3]
    ctxs = sys.argv[4].split(",")
    caller = sys.argv[5]
    transaction = sys.argv[6]

    # Remove all empty strings from tweet_ids and ctxs
    tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id != ""]
    ctxs = [ctx for ctx in ctxs if ctx != ""]

    if len(tweet_ids) > 0:
        # Save cache
        tweet_ids_cache = tweet_ids

        # First check if tweets already exist
        status, existing_tweet_ids, existing_usernames, existing_tweets = (
            check_existing_tweets(tweet_ids)
        )

        # Filter out tweets that don't exist
        tweet_ids = [
            tweet_id for tweet_id in tweet_ids if tweet_id not in existing_tweet_ids
        ]

        # Filter out null contexts
        ctxs = [ctx for ctx in ctxs if ctx != "null"]

        if not tweet_ids:
            print(f"✅ Tweets already exist")
            sys.exit(0)

        # If not found, proceed with fetching the tweet
        tweets_data = await get_tweets_by_ids(
            tweet_ids, flag, caller, transaction, ctxs
        )

        if not tweets_data:
            print(f"❌ Failed to fetch tweets")
            sys.exit(0)

        for tweet_data in tweets_data:
            if username != "null":
                if tweet_data["username"] != username:
                    print(
                        f"⚠️  Username mismatch: {tweet_data['username']} != {username}"
                    )
                    continue

            # Create directory if it doesn't exist
            output_dir = f"tweets/{tweet_data['username']}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/input.json"

            # Save to file
            tweet_list = []
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    tweet_list = json.load(f)

            tweet_list.append(tweet_data)

            with open(output_file, "w") as f:
                json.dump(tweet_list, f, indent=2)

            # Add to database
            add_to_database(tweet_data)

            # Add to score
            add_to_score(tweet_data["username"], len(tweet_list), 0, 0, 1, ctxs)

        print(
            f"✅ ({len(tweets_data)}/{len(tweet_ids_cache)}) tweets saved to {output_file}"
        )
    elif len(tweet_ids) == 0 and username != "@" and len(ctxs) > 0:
        # Filter out null contexts
        ctxs = [ctx for ctx in ctxs if ctx != "null"]
        # Get tweets by username
        output_dir = f"tweets/{username}"
        output_file = f"{output_dir}/input.json"
        tweet_list = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                tweet_list = json.load(f)
        # Add to score
        add_to_score(username, len(tweet_list), 0, 0, 1, ctxs)


if __name__ == "__main__":
    asyncio.run(main())
    print(f"✅ Finished processing tweets")
