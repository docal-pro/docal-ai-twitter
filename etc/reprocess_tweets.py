import json
import csv
from typing import Dict, List
import os


def analyse_tweet_contexts(user, function):
    """Analyse tweets to count replies and check for contexts that match tweet text."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load tweet data from tweets/{user}/input.json
    with open(os.path.join(parent_dir, f"tweets/{user}/input.json"), "r") as f:
        all_tweets = json.load(f)

    # Load results data
    with open(os.path.join(parent_dir, f"results/{user}/{function}.csv"), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        results = list(reader)

    # Create mapping of tweet IDs to their results
    results_map = {row[0]: row for row in results}

    reply_count = 0
    true_contexts = 0
    user_prefix_contexts = 0
    other_contexts = 0

    print("\nAnalysing tweet contexts...")
    for tweet in all_tweets:
        tweet_id = tweet["id"]
        tweet_text = tweet["text"].strip()

        # Check if it's a reply (starts with @)
        is_reply = tweet_text.startswith("@")
        if is_reply:
            reply_count += 1

            # Get corresponding result
            if tweet_id in results_map:
                result_row = results_map[tweet_id]
                text = (
                    result_row[1].strip() if len(result_row) > 1 else ""
                )  # Text from results
                context = (
                    result_row[3].strip() if len(result_row) > 3 else ""
                )  # Context from results

                # Remove username from tweet text
                text_parts = tweet_text.split(" ", 1)
                actual_message = text_parts[1] if len(text_parts) > 1 else ""

                # Check different context patterns
                if context == "TRUE":
                    true_contexts += 1
                    if true_contexts <= 5:
                        print(f"\nTRUE Context Example #{tweet_id}:")
                        print(f"Text: {tweet_text}")
                        print(f"Context: {context}")
                elif context.startswith("@{user}: "):
                    user_prefix_contexts += 1
                    if user_prefix_contexts <= 10:
                        print(f"\nUser Prefix Example #{tweet_id}:")
                        print(f"Text: {tweet_text}")
                        print(f"Context: {context}")
                else:
                    other_contexts += 1
                    if other_contexts <= 10:
                        print(f"\nOther Context Example #{tweet_id}:")
                        print(f"Text: {tweet_text}")
                        print(f"Context: {context}")

    print(f"\nAnalysis Results:")
    print(f"Total replies: {reply_count}")
    print(
        f"Contexts marked as TRUE: {true_contexts} ({(true_contexts/reply_count*100):.1f}%)"
    )
    print(
        f"Contexts with @ prefix: {user_prefix_contexts} ({(user_prefix_contexts/reply_count*100):.1f}%)"
    )
    print(f"Other contexts: {other_contexts} ({(other_contexts/reply_count*100):.1f}%)")


if __name__ == "__main__":
    analyse_tweet_contexts()
