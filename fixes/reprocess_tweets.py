import json
import csv
import asyncio
from frank_context import TweetContextBuilder
import sys
import os
from typing import Dict, List, Tuple
import time
from datetime import datetime

def verify_affected_tweets(results_file: str) -> set:
    """Verify tweets that need reprocessing by checking if context is '@frankdegods: ' + text content (without username)"""
    affected_ids = set()
    print("\nDebug: Reading results.csv...")
    with open(results_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Debug: Header is {header}")
        count = 0
        for row in reader:
            count += 1
            if len(row) >= 4:  # Ensure we have text and context fields
                text = row[1]  # Changed from row[2] to row[1] for text
                context = row[3]
                print(f"\nDebug: Row {count}")
                print(f"Text: '{text}'")
                print(f"Context: '{context}'")
                
                # Extract the actual message without the username
                text_parts = text.strip().split(' ', 1)
                if len(text_parts) > 1:
                    username = text_parts[0]
                    actual_message = text_parts[1]
                    expected_context = f"@frankdegods: {actual_message}"
                    print(f"Expected context: '{expected_context}'")
                    if context.strip() == expected_context:
                        affected_ids.add(row[0])  # Add created_at as identifier
                        print(f"MATCH FOUND!")
            if count >= 10:  # Just look at first 10 rows for debug
                break
    return affected_ids

def save_progress(processed_tweets: List[List], header: List[str], backup_suffix: str):
    """Save current progress to a temporary file"""
    progress_file = f'results_in_progress_{backup_suffix}.csv'
    with open(progress_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(processed_tweets)
    print(f"\nProgress saved to {progress_file}")

async def process_single_tweet(tweet_id: str, tweet_data: Dict, context_builder: TweetContextBuilder, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        try:
            # Get the original text without any username prefix
            text = tweet_data['text']
            if text.startswith('@'):
                # Remove the username and any following whitespace
                text = text.split(' ', 1)[1] if ' ' in text else ''
            
            # Set the context to be the cleaned text without the @frankdegods prefix
            tweet_data['context'] = text
            
            # Use the context builder to process the tweet
            await context_builder.build_context_thread(tweet_data)
        except Exception as e:
            print(f"Error processing tweet {tweet_id}: {str(e)}")

async def reprocess_tweets():
    # Load affected tweet IDs
    with open('affected_tweet_ids.txt', 'r') as f:
        affected_ids = set(line.strip() for line in f)
    
    print(f"Total tweets to reprocess: {len(affected_ids)}")
    
    # Load tweet data from frank_tweets.json
    with open('frank_tweets.json', 'r') as f:
        all_tweets = json.load(f)
    
    # Create mapping of tweet IDs to their data
    tweets_data = {tweet['id']: tweet for tweet in all_tweets}
    
    # Initialize the context builder with required file paths
    context_builder = TweetContextBuilder('frank_tweets.json', 'results.csv')
    
    # Create a semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(5)
    
    # Create tasks for each affected tweet
    tasks = []
    processed_count = 0
    
    for tweet_id in affected_ids:
        if tweet_id in tweets_data:
            task = asyncio.create_task(process_single_tweet(tweet_id, tweets_data[tweet_id], context_builder, semaphore))
            tasks.append(task)
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Progress: {processed_count}/{len(affected_ids)} tweets ({(processed_count/len(affected_ids)*100):.1f}%)")
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"Completed reprocessing {len(affected_ids)} tweets")

if __name__ == "__main__":
    asyncio.run(reprocess_tweets()) 