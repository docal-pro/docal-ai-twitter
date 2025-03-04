from datetime import datetime
import json
import os
import sys
import asyncio
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from utils.utils import connect_to_database, add_to_database, add_to_score

# Load environment variables
load_dotenv()


async def get_tweets(tweet_ids: list[str], flag: str = "none"):
    """Get tweet data using agent-twitter-client"""
    print(f"🔎 Searching for tweets...")
    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            "agent/scraper.js",
            ",".join(tweet_ids),
            flag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        print(f"ℹ️  Waiting for tweets...")
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if stderr:
            stderr_text = stderr.decode()
            print(f"❌ Error: {stderr_text}")
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
            print(f"❌ No tweets found")
            return []
        
        return tweets_data

    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return []
    

def check_existing_tweets(tweet_ids: list[str]) -> tuple[bool, str, list]:
    """
    Check if tweets already exist in the database
    Returns: (exists: bool, username: str if exists else None, existing_tweets: list)
    """
    print(f"🔎 Checking for tweets...")
    
    # Query the database for the tweet IDs
    try:
        connection, cursor = connect_to_database()

        # Query for tweets
        placeholders = ','.join(['%s'] * len(tweet_ids))
        cursor.execute(
            f"SELECT tweet_id, username, tweet FROM twitter WHERE tweet_id IN ({placeholders})",
            tuple(tweet_ids)
        )
        results = cursor.fetchall()

        if results:
            matched_tweet_ids = [row[0] for row in results]
            matched_usernames = [row[1] for row in results]
            matched_tweets = [row[2] for row in results]
            return True, matched_tweet_ids, matched_usernames, matched_tweets
            
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        
    return False, [], [], []


async def main():
    if len(sys.argv) != 5:
        print("ℹ️  Usage: python scraper.py <tweet_ids> <username> <flag> <contexts>")
        sys.exit(1)

    # Parse comma-separated tweet IDs
    tweet_ids = sys.argv[1].split(',')
    username = sys.argv[2]
    flag = sys.argv[3]
    ctxs = sys.argv[4].split(',')
    
    # First check if tweets already exist
    status, existing_tweet_ids, existing_usernames, existing_tweets = check_existing_tweets(tweet_ids)

    # Filter out tweets that don't exist
    tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in existing_tweet_ids]

    if not tweet_ids:
        print(f"✅ Tweets already exist")
        sys.exit(0)

    # If not found, proceed with fetching the tweet
    tweets_data = await get_tweets(tweet_ids, flag)
    
    if not tweets_data:
        print(f"❌ No tweets found")
        sys.exit(1)
    
    for tweet_data in tweets_data:
        # Check if username (if not null) is the same as the one in the tweet
        if username != "null":
            if tweet_data['username'] != username:
                print(f"⚠️  Username mismatch: {tweet_data['username']} != {username}. Skipping...")
                continue

        # Create directory if it doesn't exist
        output_dir = f"tweets/{tweet_data['username']}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/input.json"
        
        # Save to file
        tweet_list = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                tweet_list = json.load(f)
        
        tweet_list.append(tweet_data)
        
        with open(output_file, 'w') as f:
            json.dump(tweet_list, f, indent=2)
            
        print(f"✅ Tweet saved to {output_file}")
            
        # Add to database
        add_to_database(tweet_data)
        
        # Add to score
        add_to_score(tweet_data['username'], len(tweet_list), 0, 0, 1, ctxs)
        


if __name__ == "__main__":
    asyncio.run(main())
    print(f"✅ Finished processing tweets")
