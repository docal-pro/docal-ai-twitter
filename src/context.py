import json
import csv
import os
import sys
import time
import shutil
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import asyncio
from pyppeteer import launch
import subprocess
from datetime import datetime

# Load arguments
user = sys.argv[1]

# Load environment variables from .env file
load_dotenv()


def create_backup(file_path: str, backup_dir: str = f"results/backups/{user}") -> str:
    """Create a backup of a file with timestamp"""
    if not os.path.exists(file_path):
        return None

    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(file_path)
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")

    # Create the backup
    shutil.copy2(file_path, backup_path)
    print(f"📁 Created backup: {backup_path}")
    return backup_path


class TweetContextBuilder:
    def __init__(self, input_file, output_file, limit=None):
        self.input_file = input_file
        self.output_file = output_file
        self.limit = limit
        self.browser = None
        self.page = None

        # Load tweets at initialisation
        with open(input_file, "r") as f:
            self.tweets = json.load(f)

        # Initialise stats based on current progress
        self.stats = {
            "processed_tweets": 0,
            "successful_contexts": 0,
            "failed_contexts": 0,
            "timeout_errors": 0,
            "other_errors": 0,
            "api_errors": 0,
            "rate_limits_hit": 0,
            "restricted_tweets": 0,
            "status": " Running ⏯",
            "error_status": " Active ✓",
        }

        # Load current progress from output file
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    self.stats["processed_tweets"] += 1
                    if len(row) > 4:  # Check if context column exists
                        if "[This post is unavailable]" in row[4]:
                            self.stats["failed_contexts"] += 1
                        elif "[This post was not accessible" in row[4]:
                            self.stats["failed_contexts"] += 1
                            self.stats["api_errors"] += 1
                        elif "[This post is restricted]" in row[4]:
                            self.stats["restricted_tweets"] += 1
                        elif "[This post was not accessible due to timeout]" in row[4]:
                            self.stats["timeout_errors"] += 1
                        else:
                            self.stats["successful_contexts"] += 1

        # Configuration
        self.config = {
            "max_retries": 5,
            "retry_delay": 5000,  # ms
            "min_delay": 1000,  # ms
            "max_delay": 3000,  # ms
            "rate_limit_threshold": 3,
        }

    def print_stats(self):
        """Print current processing stats"""
        total = len(self.tweets)
        status = self.stats["status"]
        error_status = self.stats["error_status"]
        processed = self.stats["processed_tweets"]
        error_count = (
            self.stats["failed_contexts"]
            + self.stats["timeout_errors"]
            + self.stats["api_errors"]
            + self.stats["restricted_tweets"]
        )

        progress = (processed / total) * 100 if total > 0 else 0
        error_percentage = (error_count / processed * 100) if processed > 0 else 0

        print("\033[H\033[J")  # Clear screen
        print("=== Tweet Processing Status ===")
        print("┌─────────────┬───────────────┬────────────┬────────────┐")
        print("│ Metric      │ Count         │ Status     │ % of Total │")
        print("├─────────────┼───────────────┼────────────┼────────────┤")
        print(
            f"│ Progress    │ {processed}/{total:<11,} │{status}  │ {progress:>6.1f}%    │"
        )
        print("├─────────────┼───────────────┼────────────┼────────────┤")
        print(
            f"│ Errors      │ {error_count:<13,} │ {error_status}  │ {error_percentage:>6.1f}%    │"
        )
        print("└─────────────┴───────────────┴────────────┴────────────┘")
        print()

    def init_stats(self):
        """Initialise stats from historical data"""
        if os.path.exists(self.output_file):
            try:
                print("\n🔎 Analysing existing output file...")
                with open(self.output_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    processed = sum(1 for _ in reader)
                    self.stats["processed_tweets"] = processed
                print(f"✅ Found {processed:,} processed tweets")
                self.print_stats()
            except Exception as e:
                print(f"❌ Error loading historical stats: {e}")
                import traceback

                traceback.print_exc()

    async def init_browser(self):
        """Initialise the browser"""
        if not self.browser:
            self.browser = await launch(
                headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            self.page = await self.browser.newPage()
            await self.page.setUserAgent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )

    async def close_browser(self):
        """Close the browser"""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None

    def handle_circular_refs(self, obj):
        seen = set()

        def serialise(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return "[Circular]"
            seen.add(obj_id)
            if isinstance(obj, dict):
                return {k: serialise(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialise(item) for item in obj]
            return obj

        return serialise(obj)

    async def get_tweet_with_agent(self, tweet_id: str) -> Optional[Dict]:
        """Try to get tweet using agent-twitter-client"""
        print(f"\033[2J\033[H")  # Clear screen and move cursor to top
        self.print_stats()  # Show updated stats
        print(f"\n🔎 Getting tweet {tweet_id}...")

        try:
            process = await asyncio.create_subprocess_exec(
                "node",
                "agent/tweet.js",
                tweet_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
            except asyncio.TimeoutError:
                process.kill()
                self.stats["timeout_errors"] += 1
                self.print_stats()  # Update stats display immediately
                return {
                    "id": tweet_id,
                    "failed": True,
                    "error_type": "timeout",
                    "text": "[This post was not accessible due to timeout]",
                    "username": "unknown",
                }

            if stderr:
                stderr_text = stderr.decode()
                if "Missing data" in stderr_text:
                    self.stats["restricted_tweets"] += 1
                    self.print_stats()  # Update stats display immediately
                    return {
                        "id": tweet_id,
                        "failed": True,
                        "error_type": "restricted",
                        "text": "[This post is restricted]",
                        "username": "restricted",
                    }
                elif "Not authorized" in stderr_text:
                    self.stats["restricted_tweets"] += 1
                    self.print_stats()  # Update stats display immediately
                    return {
                        "id": tweet_id,
                        "failed": True,
                        "error_type": "suspended",
                        "text": "[This post is from a suspended account.]",
                        "username": "suspended",
                    }
                elif "deleted" in stderr_text.lower():
                    self.stats["api_errors"] += 1
                    self.print_stats()  # Update stats display immediately
                    return {
                        "id": tweet_id,
                        "failed": True,
                        "error_type": "deleted",
                        "text": "[This post was deleted]",
                        "username": "deleted",
                    }
                self.stats["api_errors"] += 1
                self.print_stats()  # Update stats display immediately
                return {
                    "id": tweet_id,
                    "failed": True,
                    "error_type": "api_error",
                    "text": "[This post was not accessible due to an API error]",
                    "username": "unknown",
                }

            try:
                tweet_data = stdout.decode().strip()
                if not tweet_data:
                    self.stats["api_errors"] += 1
                    self.print_stats()  # Update stats display immediately
                    return {
                        "id": tweet_id,
                        "failed": True,
                        "error_type": "empty",
                        "text": "[This post was not accessible]",
                        "username": "unknown",
                    }
                tweet_json = json.loads(tweet_data)
                return tweet_json
            except json.JSONDecodeError as e:
                self.stats["api_errors"] += 1
                self.print_stats()  # Update stats display immediately
                return {
                    "id": tweet_id,
                    "failed": True,
                    "error_type": "invalid_data",
                    "text": "[This post was not accessible due to invalid data]",
                    "username": "unknown",
                }
        except Exception as e:
            self.stats["other_errors"] += 1
            self.print_stats()  # Update stats display immediately
            return {
                "id": tweet_id,
                "failed": True,
                "error_type": "unknown",
                "text": "[This post was not accessible due to an unknown error]",
                "username": "unknown",
            }

    async def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict]:
        """Get a tweet by its ID using agent-twitter-client"""
        tweet_data = await self.get_tweet_with_agent(tweet_id)
        if tweet_data:
            return tweet_data
        print(f"❌ Failed to get tweet {tweet_id}")
        return None

    async def build_context_thread(self, tweet: Dict) -> str:
        """Build the conversation thread context by recursively getting all parent tweets"""
        max_retries_per_tweet = 3  # Maximum number of retries per tweet
        thread_tweets = []
        visited_ids = set()
        tweets_by_id = {}
        failed_to_get_parent = False
        error_label = ""

        # First, load the tweet into our cache
        tweets_by_id[tweet["id"]] = tweet
        thread_tweets.append(tweet)

        async def get_parent_tweet(tweet_id: str) -> Optional[Dict]:
            """Get a parent tweet either from our dataset or by scraping"""
            retry_count = 0
            while retry_count < max_retries_per_tweet:
                try:
                    # First check if we have this tweet in our local cache
                    if tweet_id in tweets_by_id:
                        return tweets_by_id[tweet_id]

                    # Try to get the tweet using our dual approach
                    print(
                        f"🔎 Getting tweet {tweet_id} (attempt {retry_count + 1}/{max_retries_per_tweet})..."
                    )
                    parent_tweet = await asyncio.wait_for(
                        self.get_tweet_by_id(tweet_id),
                        timeout=45,  # 45 second timeout for entire operation
                    )
                    if parent_tweet:
                        if parent_tweet.get("failed", False):
                            error_type = parent_tweet.get("error_type", "unknown")
                            if error_type == "suspended":
                                self.stats["restricted_tweets"] += 1
                                return {
                                    "error_label": "[This post is from a suspended account.]"
                                }
                            elif error_type == "restricted":
                                self.stats["restricted_tweets"] += 1
                                return {"error_label": "[This post is restricted]"}
                            elif error_type == "deleted":
                                self.stats["api_errors"] += 1
                                return {"error_label": "[This post was deleted]"}
                            elif error_type == "timeout":
                                self.stats["timeout_errors"] += 1
                                return {
                                    "error_label": "[This post was not accessible due to timeout]"
                                }
                            else:
                                self.stats["other_errors"] += 1
                                return {
                                    "error_label": parent_tweet.get(
                                        "text", "[This post was not accessible]"
                                    )
                                }
                            self.print_stats()
                        tweets_by_id[tweet_id] = parent_tweet
                        thread_tweets.append(parent_tweet)
                        return parent_tweet
                except asyncio.TimeoutError:
                    print(f"❌ Timeout on attempt {retry_count + 1} for tweet {tweet_id}")
                    self.stats["timeout_errors"] += 1
                    self.print_stats()
                except Exception as e:
                    print(
                        f"❌ Error on attempt {retry_count + 1} for tweet {tweet_id}: {str(e)}"
                    )
                    self.stats["other_errors"] += 1
                    self.print_stats()

                retry_count += 1
                if retry_count < max_retries_per_tweet:
                    await asyncio.sleep(5)  # Wait 5 seconds between retries

            print(
                f"❌ Failed to get tweet {tweet_id} after {max_retries_per_tweet} attempts"
            )
            self.stats["failed_contexts"] += 1
            self.print_stats()
            return {
                "error_label": "[This post was not accessible after multiple attempts]"
            }

        # Start with the current tweet
        current_tweet = tweet
        current_id = tweet["id"]

        while True:
            # Add current tweet ID to visited set
            visited_ids.add(current_id)

            # Get the parent tweet ID
            parent_id = None
            if (
                "inReplyToStatusId" in current_tweet
                and current_tweet["inReplyToStatusId"]
            ):
                parent_id = current_tweet["inReplyToStatusId"]

            # Break if no parent or we've seen this ID before
            if not parent_id or parent_id in visited_ids:
                break

            # Try to get the parent tweet with exponential backoff
            parent_tweet = None
            for attempt in range(self.config["max_retries"]):
                parent_tweet = await get_parent_tweet(parent_id)
                if parent_tweet:
                    if "error_label" in parent_tweet:
                        error_label = parent_tweet["error_label"]
                        failed_to_get_parent = True
                        break
                    current_tweet = parent_tweet
                    current_id = parent_id
                    break

                # If we hit rate limit, wait longer
                if self.stats["rate_limits_hit"] > self.config["rate_limit_threshold"]:
                    delay = min(
                        self.config["retry_delay"] * (2**attempt), 30000
                    )  # Max 30 seconds
                else:
                    delay = self.config["min_delay"]

                print(f"🔄 Retrying after {delay}ms...")
                await asyncio.sleep(delay / 1000)  # Convert to seconds

            if (
                not parent_tweet or failed_to_get_parent
            ):  # If we couldn't get the parent tweet after all retries
                break

        # Now format the thread in chronological order (oldest to newest)
        thread_tweets.reverse()  # Reverse to get chronological order
        formatted_thread = []

        # Add error label if we had one
        if error_label:
            formatted_thread.append(error_label)

        for t in thread_tweets:
            username = t.get("username", "")
            text = t.get("text", "")
            # Remove the @username from the start of reply tweets
            if text.startswith("@"):
                text = " ".join(text.split()[1:])
            formatted_thread.append(f"@{username}: {text}")

        return "\n".join(formatted_thread)

    async def process_tweets(self):
        """Process all tweets and generate CSV output"""
        with open(self.input_file, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        # Limit the number of tweets to process
        tweets = tweets[: self.limit]
        print(f"ℹ️  Processing {len(tweets)} tweets...")

        # Always append to preserve history
        file_exists = os.path.exists(self.output_file)
        mode = "a" if file_exists else "w"

        with open(self.output_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:  # Only write header for new file
                writer.writerow(["tweet_id", "createdAt", "text", "isReply", "context"])

            for tweet in tweets:
                try:
                    # Get thread context
                    context = await self.build_context_thread(tweet)

                    # Write row with tweet_id first
                    writer.writerow(
                        [
                            tweet["id"],  # Add tweet_id as first column
                            tweet["createdAt"],
                            tweet["text"],
                            "True" if tweet.get("isReply", False) else "False",
                            context,
                        ]
                    )

                    # Update stats
                    self.stats["processed_tweets"] += 1
                    if "[Failed to fetch tweet" not in context:
                        self.stats["successful_contexts"] += 1

                    # Print progress
                    self.print_stats()

                except Exception as e:
                    print(f"❌ Error processing tweet: {str(e)}")
                    self.stats["other_errors"] += 1
                    continue

    def load_historical_stats(self):
        """Load historical stats from existing output file"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    self.stats["processed_tweets"] = sum(1 for _ in reader)
                self.print_stats()
            except Exception as e:
                print(f"❌ Error loading historical stats: {e}")
                import traceback

                traceback.print_exc()


async def main():
    input_file = f"tweets/{user}/input.json"
    output_file = f"results/{user}/context.csv"
    backup_dir = f"results/{user}/backups"
    checkpoints_dir = f"results/{user}/checkpoints"
    checkpoint_file = f"{checkpoints_dir}/context_checkpoint.csv"
    processed_ids_file = f"{checkpoints_dir}/context_processed.json"
    batch_size = 5  # Process tweets in batches

    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Load all tweets
    with open(input_file, "r", encoding="utf-8") as f:
        all_tweets = json.load(f)

    # Load previously processed tweet IDs if they exist
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 0:
                    processed_ids.add(row[0])  # First column is tweet_id

    # Create output file with header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tweet_id", "createdAt", "text", "isReply", "context"])

    # Initialise processor with current progress
    processor = TweetContextBuilder(input_file, output_file, limit=None)

    # Filter out already processed tweets
    tweets_to_process = [t for t in all_tweets if t["id"] not in processed_ids]
    print(f"\n🔎 Resuming with {len(tweets_to_process)} tweets left to process")

    # Process tweets in batches
    for i in range(0, len(tweets_to_process), batch_size):
        batch = tweets_to_process[i : i + batch_size]
        print(
            f"\n🔎 Processing batch {i//batch_size + 1}/{len(tweets_to_process)//batch_size + 1}"
        )

        for tweet in batch:
            try:
                # Add timeout for entire tweet processing
                context = await asyncio.wait_for(
                    processor.build_context_thread(tweet),
                    timeout=30,  # 30 second timeout per tweet
                )

                # Append to main CSV
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            tweet["id"],
                            tweet["createdAt"],
                            tweet["text"],
                            "True" if tweet.get("isReply", False) else "False",
                            context,
                        ]
                    )

                # Update processed IDs and save immediately
                processed_ids.add(tweet["id"])
                with open(processed_ids_file, "w") as f:
                    json.dump(list(processed_ids), f)

                # Update processed count and stats
                processor.stats["processed_tweets"] += 1
                if "[This post is unavailable]" in context:
                    processor.stats["failed_contexts"] += 1
                elif "[This post was not accessible" in context:
                    processor.stats["failed_contexts"] += 1
                    processor.stats["api_errors"] += 1
                elif "[This post is restricted]" in context:
                    processor.stats["restricted_tweets"] += 1
                elif "[This post was not accessible due to timeout]" in context:
                    processor.stats["timeout_errors"] += 1
                else:
                    processor.stats["successful_contexts"] += 1

                processor.print_stats()

                # Create backup after each tweet
                backup_path = create_backup(output_file)
                if backup_path:
                    subprocess.run(["cp", backup_path, checkpoint_file])

                # Small delay between tweets
                await asyncio.sleep(0.5)

            except asyncio.TimeoutError:
                print(f"❌ Timeout processing tweet {tweet['id']}, skipping...")
                processor.stats["timeout_errors"] += 1
                continue
            except Exception as e:
                print(f"❌ Error processing tweet {tweet['id']}: {str(e)}")
                processor.stats["other_errors"] += 1
                continue

        # Add delay between batches
        await asyncio.sleep(2)

    processor.stats["status"] = " Done ✓   "
    processor.stats["error_status"] = "None ✓   "
    processor.print_stats()
    processor.stats["status"] = "Complete"
    processor.stats["error_status"] = "None"
    print("\n✅ Processing complete! Stats:")
    for key, value in processor.stats.items():
        print(f"{key}: {value}")

    await processor.close_browser()


if __name__ == "__main__":
    asyncio.run(main())
