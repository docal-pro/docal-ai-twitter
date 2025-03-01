import json
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def get_tweet(tweet_id: str):
    """Get tweet data using agent-twitter-client"""
    print(f"🔎 Searching for tweet {tweet_id}...")
    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            "agent/agent.js",
            tweet_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        print(f"ℹ️  Waiting for tweet {tweet_id}...")
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if stderr:
            stderr_text = stderr.decode()
            print(f"❌ Error: {stderr_text}")
            error_message = "[This post is not accessible]"
            if "Missing data" in stderr_text:
                error_message = "[This post is restricted]"
            elif "Not authorized" in stderr_text:
                error_message = "[This post is from a suspended account]"
            elif "deleted" in stderr_text.lower():
                error_message = "[This post was deleted]"
            
            return {
                "id": tweet_id,
                "failed": True,
                "text": error_message,
                "username": "unknown"
            }

        # Step 1: Decode the bytes to a string
        decoded_data = stdout.decode("utf-8")

        # Step 2: Split logs and JSON
        lines = decoded_data.split("\n")
        logs = []
        json_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{"):
                json_lines.append(stripped)
            elif json_lines:  # If JSON started, assume multi-line JSON
                json_lines.append(stripped)
            else:
                logs.append(stripped)

        # Step 3: Print logs
        for log in logs:
            print(log)

        # Step 4: Parse and print JSON
        json_string = "".join(json_lines) if json_lines else "{}"  # Ensure json_string is always defined

        try:
            tweet_data = json.loads(json_string)
            # print(json.dumps(tweet_data, indent=2))  # Pretty-print JSON
        except json.JSONDecodeError as e:
            print("\n❌ Error parsing JSON:", e)
            tweet_data = None  # Ensure `tweet_data` is defined

        if not tweet_data:
            return {
                "id": tweet_id,
                "failed": True,
                "text": "[This post was not accessible]",
                "username": "unknown"
            }
        
        return tweet_data

    except Exception as e:
        print(f"❌ Error fetching tweet: {e}")
        return {
            "id": tweet_id,
            "failed": True,
            "text": "[Error fetching tweet]",
            "username": "unknown"
        }


def check_existing_tweet(tweet_id: str) -> tuple[bool, str, list]:
    """
    Check if tweet already exists in any user's input.json file
    Returns: (exists: bool, username: str if exists else None, existing_tweets: list)
    """
    # Check all subdirectories in tweets/
    tweets_dir = Path("tweets")
    if not tweets_dir.exists():
        return False, None, []
    
    for user_dir in tweets_dir.iterdir():
        if not user_dir.is_dir():
            continue
            
        input_file = user_dir / "input.json"
        if not input_file.exists():
            continue
            
        try:
            with open(input_file, 'r') as f:
                tweets = json.load(f)
                # Check if tweet_id exists in this file
                for tweet in tweets:
                    if tweet.get('id') == tweet_id:
                        return True, user_dir.name, tweets
        except json.JSONDecodeError:
            continue
    
    return False, None, []


async def main():
    if len(sys.argv) != 2:
        print("ℹ️  Usage: python tweet_scraper.py <tweet_id>")
        sys.exit(1)

    tweet_id = sys.argv[1]
    
    # First check if tweet already exists
    exists, existing_username, existing_tweets = check_existing_tweet(tweet_id)
    if exists:
        print(f"✅ Tweet {tweet_id} already exists in tweets/{existing_username}/input.json")
        sys.exit(0)
    
    # If not found, proceed with fetching the tweet
    tweet_data = await get_tweet(tweet_id)
    
    if tweet_data.get("failed"):
        print("❌ Failed to fetch tweet")
        sys.exit(1)
        
    username = tweet_data.get("username")
    if not username:
        print("❌ Could not extract username from tweet")
        sys.exit(1)

    # Create directory if it doesn't exist
    output_dir = f"tweets/{username}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/input.json"
    
    # Save to file
    tweets_list = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            tweets_list = json.load(f)
    
    tweets_list.append(tweet_data)
    
    with open(output_file, 'w') as f:
        json.dump(tweets_list, f, indent=2)
    
    print(f"✅ Tweet saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
