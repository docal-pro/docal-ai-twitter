import os
import json
import sys
from twitter_scraper import get_tweets

def scrape_tweet(tweet_id):
    for tweet in get_tweets("", pages=1):
        if tweet["tweetId"] == tweet_id:
            username = tweet["username"]
            directory = os.path.join("tweets", username)
            os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
            output_file = os.path.join(directory, "output.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tweet, f, indent=4, ensure_ascii=False)
            print(f"✅ Tweet saved to {output_file}")
            return
    print("❌ Tweet not found")

if __name__ == "__main__":
    tweet_id = sys.argv[1]
    scrape_tweet(tweet_id)
