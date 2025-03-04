import os
import pandas as pd
import requests
import asyncio
import concurrent.futures
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import time
import sys
from collections import defaultdict
import fcntl
import csv

# Load arguments
user = sys.argv[1]
ctxs = sys.argv[2]  # Comma-separated list of contexts to classify

# Load environment variables
load_dotenv()

# Initialise API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def load_contexts(ctxs: str) -> List[str]:
    """Load the contexts from the input"""
    return ctxs.split(",")

contexts = load_contexts(ctxs)

def create_system_prompt(user: str, contexts: List[str]) -> str:
    """Create system prompt based on user and contexts"""
    prompt = f"""You are analysing tweets from @{user}. Your task is to classify the tweet according to multiple criteria.

The criteria you will evaluate are: {', '.join(contexts)}

For each criterion, provide a binary classification (Yes/No) and brief reasoning.

Return your response in this exact format for each criterion:
"""
    # Add format example for each context
    for ctx in contexts:
        prompt += f"{ctx.upper()}: [Yes or No]\nREASONING: [1-2 sentences explaining why]\n\n"
    
    return prompt.strip()

SYSTEM_PROMPT = create_system_prompt(user, contexts)

# Global progress tracking
model_progress = defaultdict(
    lambda: {"processed": 0, "total": 0, "status": "waiting", "text": " Waiting 𝌗", "last_error": None}
)

# Global file lock for saving progress
file_lock = asyncio.Lock()

def load_checkpoint(output_file: str) -> int:
    """Load the checkpoint by counting data lines in the output CSV file"""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            # Subtract 1 to account for header line
            return sum(1 for line in f) - 1
    return 0

def clear_terminal():
    """Clear the terminal screen"""
    os.system("cls" if os.name == "nt" else "clear")


def print_status_table():
    """Print a clean status table of all models"""
    clear_terminal()
    print("\n=== Model Processing Status ===")
    print(
        "┌────────────┬──────────────┬───────────┬────────────────────────────────────────────────┐"
    )
    print(
        "│ Model      │ Progress     │ Status    │ Last Error                                     │"
    )
    print(
        "├────────────┼──────────────┼───────────┼────────────────────────────────────────────────┤"
    )

    for model in sorted(model_progress.keys()):
        progress = model_progress[model]
        progress_str = f"{progress['processed']}/{progress['total']}"
        status = progress["status"]
        error = progress["last_error"]
        text = progress["text"]

        # Format error message to be more readable
        if error:
            # Try to extract more meaningful part of the error from text field
            if isinstance(error, str):
                if "400 - " in error:
                    error = error.split("400 - ", 1)[1]
                if "message" in error.lower() and ":" in error:
                    error = error.split(":", 1)[1].strip()
                # Remove quotes and curly braces
                error = error.replace("{", "").replace("}", "").replace('"', "")
                error = error[:43] + "..." if len(error) > 43 else error

        error_str = error if error else ""

        # Pad model name to 10 chars, progress to 12 chars, status to 11 chars, error to 45 chars
        model_str = f"{model:10}"
        progress_str = f"{progress_str:12}"
        status_str = f"{status:11}"
        text_str = f"{text}"
        error_str = f"{error_str:46}"

        print(f"│ {model_str} │ {progress_str} │ {text_str} │ {error_str} │")

    print(
        "└────────────┴──────────────┴───────────┴────────────────────────────────────────────────┘"
    )
    sys.stdout.flush()


def retry_with_backoff(func, *args, max_retries=3, initial_delay=1):
    """Retry a function with exponential backoff"""
    for i in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if i == max_retries - 1:  # Last attempt
                raise e

            # Check for specific error types
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                # Rate limit - use longer delay
                delay = initial_delay * (2**i)
            elif "401" in error_str or "unauthorized" in error_str:
                # Auth error - no point retrying
                raise e
            elif "quota" in error_str or "credit" in error_str:
                # Quota/credit error - no point retrying
                raise e
            else:
                # Other errors - use fixed delay
                delay = initial_delay

            time.sleep(delay)
            continue


def get_deepseek_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using DeepSeek R1"""
    def _make_request():
        if not OPENROUTER_API_KEY:
            raise Exception("401 Unauthorised - DeepSeek API key not found")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek/deepseek-r1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "stream": False,
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


def get_gpt4_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using GPT-4"""
    def _make_request():
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.choices[0].message.content

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


def get_claude_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using Claude Opus"""
    def _make_request():
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise Exception("401 Unauthorised - Claude API key not found")

        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
            temperature=0.0,
        )
        content = response.content[0].text

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


def get_gemini_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using Google Gemini"""
    def _make_request():
        if not os.getenv("GOOGLE_API_KEY"):
            raise Exception("401 Unauthorised - Gemini API key not found")

        model = genai.GenerativeModel("gemini-1.5-pro")
        generation_config = genai.GenerationConfig(
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=100
        )

        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]

        response = model.generate_content(
            SYSTEM_PROMPT + "\n\nTWEET: " + context,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        if not response.text:
            raise Exception("Empty response from Gemini")

        content = response.text

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


def get_perplexity_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using Perplexity Sonar Pro"""
    def _make_request():
        if not PERPLEXITY_API_KEY:
            raise Exception("401 Unauthorised - Perplexity API key not found")

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "temperature": 0.0,
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


def get_grok_prediction(context: str) -> Dict[str, Tuple[str, str]]:
    """Get classifications using Grok 2 by OpenRouter"""
    def _make_request():
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "x-ai/grok-2-vision-1212",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "stream": False,
            "temperature": 0,
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Parse all classifications from response
        classifications = {}
        for ctx in contexts:
            ctx_upper = ctx.upper()
            if f"{ctx_upper}:" in content:
                classification = "Yes" if f"{ctx_upper}: Yes" in content else "No"
                reasoning = content.split(f"{ctx_upper}:")[1].split("REASONING:")[1].split("\n")[0].strip()
                classifications[ctx] = (classification, reasoning)
            else:
                classifications[ctx] = ("Error", "Failed to parse response")

        return classifications

    try:
        return retry_with_backoff(_make_request, max_retries=3, initial_delay=2)
    except Exception as e:
        return {ctx: ("Error", str(e)) for ctx in contexts}


async def save_progress(df: pd.DataFrame, output_file: str):
    """Save progress with file locking to prevent concurrent writes"""
    async with file_lock:
        # Create a temporary file
        temp_file = f"{output_file}.tmp"
        df.to_csv(temp_file, index=False, quoting=csv.QUOTE_ALL)
        # Atomically replace the original file
        os.replace(temp_file, output_file)


async def process_model(
    model_name: str,
    model_func,
    tweets_to_process: List[Tuple[int, str]],
    df: pd.DataFrame,
    output_file: str,
):
    """Process all tweets for a single model"""
    total_tweets = len(df)
    
    # Count completed classifications for this model
    completed = 0
    for ctx in contexts:
        col_name = f"{ctx}_{model_name}"
        if col_name in df.columns:
            completed += (~df[col_name].isna()).sum()
    completed = completed // len(contexts)  # Average across all contexts

    model_progress[model_name] = {
        "processed": completed,
        "total": total_tweets,
        "status": "running",
        "text": "Running ⏯",
        "last_error": None,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for idx, tweet in tweets_to_process:
            try:
                print(f"\n🔎 Processing tweet {idx+1} with {model_name}...")
                classifications = await asyncio.get_event_loop().run_in_executor(
                    executor, model_func, tweet
                )
                
                # Update DataFrame with classifications for each context
                for ctx in contexts:
                    classification, reasoning = classifications.get(ctx, ("Error", "Processing failed"))
                    df.at[idx, f"{ctx}_{model_name}"] = classification
                    df.at[idx, f"{ctx}_{model_name}_reasoning"] = reasoning

                model_progress[model_name]["processed"] += 1
                await save_progress(df, output_file)
                print_status_table()

            except Exception as e:
                print(f"❌ Error processing tweet with {model_name}: {str(e)}")
                model_progress[model_name]["last_error"] = str(e)
                continue

    model_progress[model_name]["status"] = "complete"
    model_progress[model_name]["text"] = "Success ✓"
    print_status_table()
    await save_progress(df, output_file)


def calculate_consensus_prediction(df: pd.DataFrame) -> pd.Series:
    """Calculate consensus prediction based on majority vote from all models"""

    models = ["gpt4", "claude", "gemini", "perplexity", "grok", "deepseek"]

    def get_consensus(row):
        # Get all missing columns
        missing_cols = [
            f"{ctx}_{model}" for ctx in contexts for model in models if f"{ctx}_{model}" not in row
        ]

        if missing_cols:
            # Mind two spaces after warning unicode
            print(f"⚠️  Missing columns: {missing_cols}")
            return None  
        
        # Get all predictions for this row
        predictions = [row[f"{ctx}_{model}"] for ctx in contexts for model in models]
        # Count valid predictions (ignore None/NaN)
        valid_predictions = [p for p in predictions if pd.notna(p)]

        if not valid_predictions:
            return None

        # Count Predictions vs Not Predictions
        prediction_count = sum(1 for p in valid_predictions if p == "Prediction")
        total_votes = len(valid_predictions)

        if total_votes == 0:
            return None

        # If more than half vote for Prediction
        if prediction_count > total_votes / 2:
            return "Prediction"
        # If less than half vote for Prediction
        elif prediction_count < total_votes / 2:
            return "Not Prediction"
        # If it's a tie, use Perplexity's vote
        else:
            perplexity_vote = row["prediction_perplexity"]
            return perplexity_vote if pd.notna(perplexity_vote) else None

    return df.apply(get_consensus, axis=1)


async def classify_tweets_async(input_file: str, output_file: str):
    """Main function to classify tweets using multiple LLMs independently"""
    # Read input file with more robust parsing
    input_df = pd.read_csv(
        input_file, quoting=csv.QUOTE_ALL, escapechar="\\", on_bad_lines="warn"
    )

    # Drop any unnamed columns from input
    unnamed_cols = [col for col in input_df.columns if "Unnamed:" in col]
    if unnamed_cols:
        print(f"➡️  Dropping unnamed columns from input: {unnamed_cols}")
        input_df = input_df.drop(columns=unnamed_cols)

    total_tweets = len(input_df)

    try:
        # Load existing progress with more robust parsing
        df = pd.read_csv(
            output_file, quoting=csv.QUOTE_ALL, escapechar="\\", on_bad_lines="warn"
        )
        print(f"\n🔎 Loaded existing progress from {output_file}")

        # Drop any unnamed columns from output
        unnamed_cols = [col for col in df.columns if "Unnamed:" in col]
        if unnamed_cols:
            print(f"➡️  Dropping unnamed columns from output: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)

        # Verify we have all the original tweets
        if len(df) != total_tweets:
            print(
                f"\n⚠️  Warning: Output file has {len(df)} tweets but input has {total_tweets}"
            )
            print("🔎 Using input file as base and copying over existing predictions...")
            # Create a new DataFrame with all input tweets
            new_df = input_df.copy()

            # Function to normalise context text
            def normalise_text(text):
                if pd.isna(text):
                    return text
                # Remove extra whitespace and normalise newlines
                return " ".join(str(text).strip().split())

            # Create a mapping key - if tweet_id exists in both, use it, otherwise use normalised context
            if "tweet_id" in df.columns and "tweet_id" in new_df.columns:
                print("🔎 Using tweet_id for mapping predictions")
                df["key"] = df["tweet_id"].astype(str)
                new_df["key"] = new_df["tweet_id"].astype(str)
            else:
                print("🔎 Using normalised context for mapping predictions")
                df["key"] = df["context"].apply(normalise_text)
                new_df["key"] = new_df["context"].apply(normalise_text)

            # Copy over existing predictions for each model
            for model in ["deepseek", "gpt4", "claude", "gemini", "perplexity", "grok"]:
                if f"{model}_prediction" in df.columns:
                    # Create a mapping using the key
                    pred_map = pd.Series(
                        df[f"{model}_prediction"].values, index=df["key"]
                    ).to_dict()
                    reason_map = pd.Series(
                        df[f"{model}_reasoning"].values, index=df["key"]
                    ).to_dict()

                    # Map the predictions/reasoning to the new DataFrame
                    new_df[f"{model}_prediction"] = new_df["key"].map(pred_map)
                    new_df[f"{model}_reasoning"] = new_df["key"].map(reason_map)

                    # Count non-null predictions
                    completed = new_df[f"{model}_prediction"].notna().sum()
                    print(f"✅ Copied {completed} predictions for {model}")

            # Remove the temporary key column
            new_df = new_df.drop("key", axis=1)
            df = new_df

            # Save the corrected DataFrame immediately
            df.to_csv(output_file, index=False)

    except FileNotFoundError:
        print(f"\n🔄 No existing progress found, starting fresh")
        df = input_df.copy()

    # Initialise consensus column if it doesn't exist
    if "consensus_prediction" not in df.columns:
        df["consensus_prediction"] = None

    # Initialise columns if they don't exist
    models = {
        "grok": get_grok_prediction,
        "deepseek": get_deepseek_prediction,
        "gpt4": get_gpt4_prediction,
        "claude": get_claude_prediction,
        "gemini": get_gemini_prediction,
        "perplexity": get_perplexity_prediction,
    }
    
    # Initialise progress for all models with total_tweets as denominator
    for model in models.keys():
        # Initialise prediction columns if they don't exist
        for ctx in contexts:
            if f"{ctx}_{model}" not in df.columns:
                df[f"{ctx}_{model}"] = None
                df[f"{ctx}_{model}_reasoning"] = None
                print(f"✅ Initialised columns for {ctx} and {model}")

        # Count how many predictions we already have
        has_prediction = ~df[f"{ctx}_{model}"].isna()
        completed = sum(has_prediction)
        print(f"✅ Found {completed} existing predictions for {ctx} and {model}")

        model_progress[f"{ctx}_{model}"] = {
            "processed": completed,
            "total": total_tweets,
            "status": "complete" if completed == total_tweets else "waiting",
            "text": "Success ✓" if completed == total_tweets else "Waiting 𝌗",
            "last_error": None,
        }

    # Calculate initial consensus for existing predictions
    df["consensus_prediction"] = calculate_consensus_prediction(df)

    # Ensure proper column ordering with tweet_id first if it exists
    if "tweet_id" in df.columns:
        cols = (
            ["tweet_id"]
            + [
                col
                for col in df.columns
                if col != "consensus_prediction" and col != "tweet_id"
            ]
            + ["consensus_prediction"]
        )
    else:
        cols = [col for col in df.columns if col != "consensus_prediction"] + [
            "consensus_prediction"
        ]
    df = df[cols]

    # Save with consensus column
    df.to_csv(output_file, index=False)

    # Load checkpoint
    checkpoint = load_checkpoint(output_file)

    print_status_table()

    # Create tasks for each model with their pending tweets
    tasks = []
    for model_name, model_func in models.items():
        # Find tweets that need this model's prediction
        needs_processing = df[f"{model_name}_prediction"].isna()
        tweets_to_process = [
            (idx, row["context"])
            for idx, row in df[needs_processing].iterrows()
            if idx >= checkpoint
        ]

        if tweets_to_process:
            print(
                f"\n🔎 Starting {model_name} with {len(tweets_to_process)} tweets to process"
            )
            tasks.append(
                process_model(
                    f"{model_name}_prediction", model_func, tweets_to_process, df, output_file
                )
            )

    if tasks:
        try:
            time.sleep(1)
            await asyncio.gather(*tasks)
        except Exception as e:
            print("\n❌ Error encountered during processing!")
            print_status_table()

            # Check if Gemini failed
            if model_progress["gemini_prediction"]["status"] == "failed":
                print("\n❌ Gemini model failed. Would you like to:")
                print("1. Retry with adjusted safety settings")
                print("2. Skip Gemini and continue with other models")
                print("3. Stop processing entirely")

                choice = input("\nEnter your choice (1-3): ")
                if choice == "1":
                    print("\n➡️  Retrying Gemini with adjusted settings...")
                    # Reset Gemini progress
                    needs_processing = df["gemini_prediction"].isna()
                    tweets_to_process = [
                        (idx, row["context"])
                        for idx, row in df[needs_processing].iterrows()
                    ]
                    model_progress["gemini_prediction"] = {
                        "processed": total_tweets - len(tweets_to_process),
                        "total": total_tweets,
                        "status": "retrying",
                        "text": "Retrying ↺",
                        "last_error": None,
                    }
                    await process_model(
                        "gemini_prediction",
                        get_gemini_prediction,
                        tweets_to_process,
                        df,
                        output_file,
                    )
                elif choice == "2":
                    print("\n⚠️  Skipping Gemini, continuing with other models...")
                    remaining_tasks = [
                        t for t in tasks if t._coro.__name__ != "process_model_gemini_prediction"
                    ]
                    await asyncio.gather(*remaining_tasks)
                else:
                    print("\n❌ Stopping all processing...")
                    return

    # Final consensus calculation and save
    df["consensus_prediction"] = calculate_consensus_prediction(df)
    # Ensure proper column ordering with tweet_id first if it exists
    if "tweet_id" in df.columns:
        cols = (
            ["tweet_id"]
            + [
                col
                for col in df.columns
                if col != "consensus_prediction" and col != "tweet_id"
            ]
            + ["consensus_prediction"]
        )
    else:
        cols = [col for col in df.columns if col != "consensus_prediction"] + [
            "consensus_prediction"
        ]
    df = df[cols]
    df.to_csv(output_file, index=False)

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    input_file = f"results/{user}/context.csv"
    output_file = f"results/{user}/classifier.csv"
    asyncio.run(classify_tweets_async(input_file, output_file))
