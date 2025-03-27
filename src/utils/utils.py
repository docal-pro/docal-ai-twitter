import os
import json
import asyncio
import psycopg2
from pathlib import Path
from datetime import datetime, timezone


def scorer(new: list[float, int], old: list[float, int]) -> float:
    """
    Calculate the updated score based on the old score and the new score.
    """
    return (new[0] * new[1] + old[0] * old[1]) / (new[1] + old[1])


def connect_to_database():
    """Connect to the database"""
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")

    connection = psycopg2.connect(
        host=db_host, database=db_name, user=db_user, password=db_pass
    )
    cursor = connection.cursor()
    return connection, cursor


def add_to_schedule(schedule_data: dict):
    """Add schedule data to the database"""
    try:
        connection, cursor = connect_to_database()

        # Check if schedule already exists for a caller
        cursor.execute(
            "SELECT * FROM schedule WHERE caller = %s", (schedule_data["caller"],)
        )
        schedule_exists = cursor.fetchone()

        if schedule_exists:
            # Update existing schedule
            cursor.execute(
                "UPDATE schedule SET username = %s, transaction = %s, contexts = %s, tweet_ids = %s, timestamp = %s WHERE caller = %s",
                (
                    schedule_data["username"],
                    schedule_data["transaction"],
                    schedule_data["contexts"],
                    schedule_data["tweet_ids"],
                    datetime.now(timezone.utc),
                    schedule_data["caller"],
                ),
            )
        else:
            # Insert new schedule
            cursor.execute(
                "INSERT INTO schedule (caller, username, transaction, contexts, tweet_ids, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    schedule_data["caller"],
                    schedule_data["username"],
                    schedule_data["transaction"],
                    schedule_data["contexts"],
                    schedule_data["tweet_ids"],
                    datetime.now(timezone.utc),
                ),
            )

        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Schedule for {schedule_data['caller']} added to database")
    except Exception as e:
        print(f"❌ Error adding schedule to database: {e}")


def add_to_database(tweet_data: dict):
    """Add tweet data to the database"""
    try:
        connection, cursor = connect_to_database()

        # Insert tweet data into database
        cursor.execute(
            "INSERT INTO twitter (tweet_id, username, tweet, timestamp) VALUES (%s, %s, %s, %s)",
            (
                tweet_data["id"],
                tweet_data["username"],
                tweet_data["text"],
                datetime.now(),
            ),
        )
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Tweet {tweet_data['id']} added to database")
    except Exception as e:
        print(f"❌ Error adding tweet to database: {e}")


def add_to_score(
    username: str,
    tweet_count: int,
    score: float,
    trust: float,
    investigate: float,
    contexts: list,
):
    """Add or update user score data in the twitter_score table"""
    try:
        connection, cursor = connect_to_database()

        # Check if user exists
        cursor.execute(f"SELECT * FROM twitter_score WHERE username = %s", (username,))
        user_exists = cursor.fetchone()

        if user_exists:
            # Update existing use
            print(user_exists[5])
            print(contexts)
            print(contexts + user_exists[5])
            print(set(contexts + user_exists[5]))
            print(list(set(contexts + user_exists[5])))

            cursor.execute(
                """
                UPDATE twitter_score 
                SET tweet_count = %s, score = %s, trust = %s, investigate = %s, contexts = %s, timestamp = %s
                WHERE username = %s
                """,
                (
                    tweet_count + user_exists[1],
                    scorer([score, tweet_count], [user_exists[2], user_exists[1]]),
                    trust + user_exists[3],
                    investigate,
                    list(set(contexts + user_exists[5])),
                    datetime.now(),
                    username,
                ),
            )
        else:
            # Insert new user
            cursor.execute(
                """
                INSERT INTO twitter_score 
                (username, tweet_count, score, trust, investigate, contexts, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    username,
                    tweet_count,
                    score,
                    trust,
                    investigate,
                    contexts,
                    datetime.now(),
                ),
            )

        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Score initialised in database for {username}")
    except Exception as e:
        print(f"❌ Error adding/updating score in database: {e}")


def check_existing_contexts(username: str) -> list[str]:
    """Check if contexts already exist in the database"""
    print(f"🔎 Checking for contexts...")

    # Query the database for the contexts
    try:
        connection, cursor = connect_to_database()

        # Query for contexts
        cursor.execute(
            f"SELECT contexts FROM twitter_score WHERE username = '{username}'"
        )
        results = cursor.fetchall()

        existing_contexts = [row[0] for row in results]
        cursor.close()
        connection.close()
        return existing_contexts
    except Exception as e:
        print(f"❌ Error querying database: {e}")
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
        placeholders = ",".join(["%s"] * len(tweet_ids))
        cursor.execute(
            f"SELECT tweet_id, username, tweet FROM twitter WHERE tweet_id IN ({placeholders})",
            tuple(tweet_ids),
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


def check_existing_username(username: str) -> bool:
    """Check if username exists"""
    tweets_dir = Path("tweets")
    if not tweets_dir.exists():
        return False

    user_dir = Path(f"tweets/{username}")
    tweet_file = user_dir / "input.json"

    if tweet_file.exists():
        # Check is user has been indexed
        return True

    return False


async def get_tweets_by_ids(
    tweet_ids: list[str],
    flag: str = "none",
    caller: str = "",
    transaction: str = "",
    ctxs: list[str] = [],
):
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
            schedule_data = {
                "caller": caller,
                "transaction": transaction,
                "username": "@",
                "tweet_ids": tweet_ids,
                "contexts": ctxs,
            }
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
            print(f"❌ No tweets found")
            return []

        # Add to schedule
        schedule_data = {
            "caller": caller,
            "transaction": transaction,
            "username": "@",
            "tweet_ids": [],
            "contexts": ctxs,
        }
        add_to_schedule(schedule_data)
        return tweets_data

    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return []


async def get_tweets_by_username(
    username: str,
    flag: str = "none",
    caller: str = "",
    transaction: str = "",
    ctxs: list[str] = [],
):
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
            "username": "@",
            "tweet_ids": [],
            "contexts": ctxs,
        }
        add_to_schedule(schedule_data)
        return tweets_data

    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return []


if __name__ != "__main__":
    __all__ = [
        "scorer",
        "add_to_database",
        "add_to_schedule",
        "add_to_score",
        "check_existing_contexts",
        "check_existing_tweets",
        "check_existing_username",
        "get_tweets_by_ids",
        "get_tweets_by_username",
    ]
