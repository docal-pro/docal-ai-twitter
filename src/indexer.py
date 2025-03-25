import json
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from utils.utils import connect_to_database, add_to_database, add_to_score

# Load environment variables
load_dotenv()

# Define how old the file should be before re-fetching (in hours)
DATA_AGE_LIMIT = int(os.getenv("DATA_AGE_LIMIT", 24))


async def get_tweets(username: str, flag: str = "none", caller: str = "", transaction: str = "", ctxs: list[str] = []):
    """Get multiple tweets from a username using agent-twitter-client"""
    print(f"🔎 Searching for tweets from @{username}...")
    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            "agent/indexer.js",
            username,
            flag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        print(f"ℹ️  Waiting for tweets from @{username}...")
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if stderr:
            stderr_text = stderr.decode()
            print(f"❌ Error: {stderr_text}")
            schedule_data = {
                "caller": caller,
                "transaction": transaction,
                "username": username,
                "tweet_ids": [],
                "contexts": ctxs,
            }
            print(schedule_data)
            add_to_schedule(schedule_data)
            return []

        # Decode stdout
        decoded_data = stdout.decode("utf-8")

        # Separate logs and JSON output
        lines = decoded_data.split("\n")
        logs, json_lines = [], []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("["):  # JSON array of tweets
                json_lines.append(stripped)
            elif json_lines:  # Handle multi-line JSON
                json_lines.append(stripped)
            else:
                logs.append(stripped)

        # Print logs
        for log in logs:
            print(log)

        # Parse JSON
        json_string = "".join(json_lines) if json_lines else "[]"

        try:
            tweets_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"\n❌ Error parsing JSON: {e}")
            return []

        if not tweets_data:
            print(f"ℹ️  No tweets found for @{username}")
            return []
        
        # Add to schedule
        schedule_data = {
            "caller": caller,
            "transaction": transaction,
            "username": "",
            "tweet_ids": [],
            "contexts": ctxs,
        }
        print(schedule_data)
        add_to_schedule(schedule_data)
        return tweets_data

    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return []


def check_existing_tweets(username: str, age_limit_hours: int) -> bool:
    """Check if tweets exist and are recent enough"""
    tweets_dir = Path("tweets")
    if not tweets_dir.exists():
        return False

    user_dir = Path(f"tweets/{username}")
    tweet_file = user_dir / "tweets.json"

    if tweet_file.exists():
        # Check file modification time
        file_mod_time = datetime.fromtimestamp(tweet_file.stat().st_mtime)
        time_diff = datetime.now() - file_mod_time

        if time_diff < timedelta(hours=age_limit_hours):
            print(f"✅ Tweets for @{username} are recent (modified {time_diff.seconds // 3600} hours ago). Skipping fetch")
            return True

        print(f"⚠️  Tweets for @{username} are outdated (modified {time_diff.seconds // 3600} hours ago). Fetching new tweets...")

    return False


async def main():
    if len(sys.argv) != 6:
        print("❌ Incorrect number of arguments")
        print("ℹ️  Usage: python indexer.py <username> <flag> <contexts> <caller> <transaction>")
        sys.exit(0)

    username = sys.argv[1]
    flag = sys.argv[2]
    ctxs = sys.argv[3].split(",")
    caller = sys.argv[5]
    transaction = sys.argv[6]

    # Check if tweets exist and are fresh
    if check_existing_tweets(username, DATA_AGE_LIMIT):
        sys.exit(0)

    # Fetch tweets
    tweets_data = await get_tweets(username, flag, caller, transaction, ctxs)
    
    if not tweets_data:
        # Add to score
        add_to_score(username, 0, 0, 0, 0, ctxs)
    else:
        # Create directory
        output_dir = f"tweets/{username}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/tweets.json"

        # Save tweets to file
        with open(output_file, 'w') as f:
            json.dump(tweets_data, f, indent=2)

        print(f"✅ Tweets saved to {output_file}")
        
        # Add to database
        for tweet in tweets_data:
            add_to_database(tweet)
        
        # Add to score
        add_to_score(username, len(tweets_data), 0, 0, 1, ctxs)


if __name__ == "__main__":
    asyncio.run(main())
