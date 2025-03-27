import os
import psycopg2
from datetime import datetime, timezone


def scorer(new: list[float, int], old: list[float, int]) -> float:
    """
    Calculate the updated score based on the old score and the new score.
    """
    return (new[0] * new[1] + old[0] * old[1]) / (new[1] + old[1])

def connect_to_database():
    """Connect to the database"""
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASSWORD')

    connection = psycopg2.connect(
        host=db_host, 
        database=db_name,
        user=db_user, 
        password=db_pass
    )
    cursor = connection.cursor()
    return connection, cursor

def add_to_schedule(schedule_data: dict):
    """Add schedule data to the database"""
    try:
        connection, cursor = connect_to_database()

        # Insert tweet data into database
        cursor.execute(
            "INSERT INTO schedule (caller, username, transaction, contexts, tweet_ids, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
            (schedule_data['caller'], schedule_data['username'], schedule_data['transaction'], schedule_data['contexts'], schedule_data['tweet_ids'], datetime.now(timezone.utc))
        )
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Schedule for {schedule_data['username']} added to database")
    except Exception as e:
        print(f"❌ Error adding schedule to database: {e}")

def add_to_database(tweet_data: dict):
    """Add tweet data to the database"""
    try:
        connection, cursor = connect_to_database()

        # Insert tweet data into database
        cursor.execute(
            "INSERT INTO twitter (tweet_id, username, tweet, timestamp) VALUES (%s, %s, %s, %s)",
            (tweet_data['id'], tweet_data['username'], tweet_data['text'], datetime.now())
        )
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Tweet {tweet_data['id']} added to database")
    except Exception as e:
        print(f"❌ Error adding tweet to database: {e}")


def add_to_score(username: str, tweet_count: int, score: float, trust: float, investigate: float, contexts: list):
    """Add or update user score data in the twitter_score table"""
    try:
        connection, cursor = connect_to_database()

        # Check if user exists
        cursor.execute(
            "SELECT * FROM twitter_score WHERE username = %s",
            (username,)
        )
        user_exists = cursor.fetchone()

        if user_exists:
            # Update existing user
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
                    list(set(contexts if contexts != ["null"] else [] + user_exists[5])),
                    datetime.now(), 
                    username
                )
            )
        else:
            # Insert new user
            cursor.execute(
                """
                INSERT INTO twitter_score 
                (username, tweet_count, score, trust, investigate, contexts, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (username, tweet_count, score, trust, investigate, contexts, datetime.now())
            )

        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Score initialised in database for {username}")
    except Exception as e:
        print(f"❌ Error adding/updating score in database: {e}")


if __name__ != "__main__":
    __all__ = ['scorer', 'add_to_database', 'add_to_schedule', 'add_to_score']
