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
            "./agent.js",
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

        tweet_data = stdout.decode().strip()
        if not tweet_data:
            return {
                "id": tweet_id,
                "failed": True,
                "text": "[This post was not accessible]",
                "username": "unknown"
            }
        
        return json.loads(tweet_data)

    except Exception as e:
        print(f"❌ Error fetching tweet: {e}")
        return {
            "id": tweet_id,
            "failed": True,
            "text": "[Error fetching tweet]",
            "username": "unknown"
        }

async def main():
    if len(sys.argv) != 2:
        print("ℹ️  Usage: python scraper.py <tweet_id>")
        sys.exit(1)

    tweet_id = sys.argv[1]
    
    # Get tweet data
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
