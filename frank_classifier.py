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

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

SYSTEM_PROMPT = """You are analyzing tweets from @frankdegods. Your task is to determine if the tweet contains a prediction about the crypto market.

Classify as 'Prediction' ONLY if @frankdegods is explicitly expressing a directional view (bullish/bearish) about:
1. Specific tokens (tradable crypto tokens like $BTC, $ETH, $HYPE)
2. Projects (twitter accounts working in web3 without tokens yet)
3. Metas (crypto narratives like AI, NFT, DeFi, Layer 1, Layer 2, Meme Coins, RWA, SocialFi, GameFi, etc.)

Examples of predictions:
- "BTC to 100k" (Prediction - clear price target)
- "Layer 2s are going to explode this year" (Prediction - directional view on L2s)
- "I'm extremely bullish on NFTs" (Prediction - directional view on NFTs)

Examples of non-predictions:
- "GM" (Not Prediction - just a greeting)
- "This project looks interesting" (Not Prediction - observation without direction)
- "What do you think about ETH?" (Not Prediction - just a question)

Return your response in this exact format:
CLASSIFICATION: [Prediction or Not Prediction]
REASONING: [1-2 sentences explaining why]"""

# Global progress tracking
model_progress = defaultdict(lambda: {
    "processed": 0, 
    "total": 0, 
    "status": "Waiting",
    "last_error": None
})

def clear_terminal():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_status_table():
    """Print a clean status table of all models"""
    clear_terminal()
    print("\n=== Model Processing Status ===")
    print("┌────────────┬──────────────┬─────────────┬─────────────────┐")
    print("│ Model      │ Progress     │ Status      │ Last Error      │")
    print("├────────────┼──────────────┼─────────────┼─────────────────┤")
    
    for model in sorted(model_progress.keys()):
        progress = model_progress[model]
        progress_str = f"{progress['processed']}/{progress['total']}"
        status = progress['status']
        error = progress['last_error']
        if error and len(error) > 14:
            error = error[:11] + "..."
        error_str = error if error else ""
        
        # Pad model name to 10 chars, progress to 12 chars, status to 11 chars, error to 15 chars
        model_str = f"{model:10}"
        progress_str = f"{progress_str:12}"
        status_str = f"{status:11}"
        error_str = f"{error_str:15}"
        
        print(f"│ {model_str} │ {progress_str} │ {status_str} │ {error_str} │")
    
    print("└────────────┴──────────────┴─────────────┴─────────────────┘")
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
                delay = initial_delay * (2 ** i)
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

def get_deepseek_prediction(context: str) -> tuple[str, str]:
    """Get prediction using DeepSeek R1"""
    def _make_request():
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise Exception("401 Unauthorized - DeepSeek API key not found")
            
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            "stream": False
        }
        response = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if not result or "choices" not in result or not result["choices"]:
            raise Exception("Empty or invalid response from DeepSeek")
            
        content = result["choices"][0]["message"]["content"]
        
        # Parse the response
        classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
        reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
        
        return classification, reasoning
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        print(f"Error with DeepSeek API after retries: {str(e)}")
        return "Error", str(e)

def get_gpt4_prediction(context: str) -> tuple[str, str]:
    """Get prediction using GPT-4"""
    def _make_request():
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.0,
            max_tokens=100
        )
        content = response.choices[0].message.content
        
        # Parse the response
        classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
        reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
        
        return classification, reasoning
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        print(f"Error with OpenAI API after retries: {str(e)}")
        return "Error", str(e)

def get_claude_prediction(context: str) -> tuple[str, str]:
    """Get prediction using Claude Opus"""
    def _make_request():
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise Exception("401 Unauthorized - Claude API key not found")
            
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": context}
            ],
            temperature=0.0
        )
        content = response.content[0].text
        
        # Parse the response
        classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
        reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
        
        return classification, reasoning
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        error_str = str(e)
        if "credit balance is too low" in error_str.lower():
            error_str = "Quota exceeded - Insufficient credit balance"
        print(f"Error with Claude API after retries: {error_str}")
        return "Error", error_str

def get_gemini_prediction(context: str) -> tuple[str, str]:
    """Get prediction using Google Gemini"""
    def _make_request():
        if not os.getenv("GOOGLE_API_KEY"):
            raise Exception("401 Unauthorized - Gemini API key not found")
            
        model = genai.GenerativeModel('gemini-pro')
        
        try:
            # Configure generation settings
            generation_config = genai.GenerationConfig(
                temperature=0.0,
                candidate_count=1,
                max_output_tokens=100
            )
            
            # Set safety settings using proper enums
            safety_settings = [
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                }
            ]
            
            # Create a structured prompt
            prompt = f"""TASK: Analyze if this tweet contains a crypto market prediction.

TWEET: {context}

FORMAT:
CLASSIFICATION: [Prediction or Not Prediction]
REASONING: [1-2 sentences explaining why]

RULES:
- Classify as 'Prediction' only if there's an explicit directional view (bullish/bearish)
- Look for predictions about tokens, projects, or crypto narratives
- General observations or questions are not predictions"""

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if not response.text:
                raise Exception("Empty response from Gemini")
            
            content = response.text
            
            # Parse the response
            if "CLASSIFICATION:" not in content or "REASONING:" not in content:
                raise Exception(f"Invalid response format from Gemini: {content[:100]}...")
                
            classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
            reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
            
            return classification, reasoning
            
        except Exception as e:
            error_msg = str(e)
            if "safety" in error_msg.lower():
                raise Exception(f"Safety error: {error_msg}")
            elif "invalid" in error_msg.lower():
                raise Exception(f"Invalid response: {error_msg}")
            else:
                raise Exception(f"Gemini error: {error_msg}")
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        error_msg = str(e)
        print(f"Error with Gemini API after retries: {error_msg}")
        return "Error", error_msg

def get_perplexity_prediction(context: str) -> tuple[str, str]:
    """Get prediction using Perplexity Sonar Pro"""
    def _make_request():
        if not PERPLEXITY_API_KEY:
            raise Exception("401 Unauthorized - Perplexity API key not found")
            
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            "temperature": 0.0
        }
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
        reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
        return classification, reasoning
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        print(f"Error with Perplexity API after retries: {str(e)}")
        return "Error", str(e)

def get_grok_prediction(context: str) -> tuple[str, str]:
    """Get prediction using Grok"""
    def _make_request():
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-2-latest",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            "stream": False,
            "temperature": 0
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        classification = "Prediction" if "CLASSIFICATION: Prediction" in content else "Not Prediction"
        reasoning = content.split("REASONING:")[1].strip() if "REASONING:" in content else "No reasoning provided"
        return classification, reasoning
    
    try:
        return retry_with_backoff(_make_request)
    except Exception as e:
        return "Error", str(e)

async def process_model(model_name: str, model_func, tweets_to_process: List[Tuple[int, str]], df: pd.DataFrame, output_file: str):
    """Process all tweets for a single model"""
    # Get the total number of tweets from the dataframe
    total_tweets = len(df)
    
    # Count how many predictions we already have
    has_prediction = ~df[f'prediction_{model_name}'].isna()
    initial_completed = sum(has_prediction)
    
    model_progress[model_name] = {
        "processed": initial_completed,
        "total": total_tweets,
        "status": "Running",
        "last_error": None
    }
    print_status_table()
    
    error_count = 0
    auth_error = False
    quota_error = False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for idx, context in tweets_to_process:
            try:
                # Skip processing if we've hit auth/quota errors
                if auth_error or quota_error:
                    continue
                    
                pred, reason = await asyncio.get_event_loop().run_in_executor(executor, model_func, context)
                
                # Skip if we got an error response
                if pred == "Error":
                    error_str = reason.lower()
                    model_progress[model_name]["last_error"] = reason
                    
                    # Check for auth/quota errors
                    if "401" in error_str or "unauthorized" in error_str:
                        auth_error = True
                        model_progress[model_name]["status"] = "Auth ❌"
                        print_status_table()
                        return
                    elif "quota" in error_str or "credit" in error_str:
                        quota_error = True
                        model_progress[model_name]["status"] = "Quota ❌"
                        print_status_table()
                        return
                    
                    error_count += 1
                    if error_count >= 5:  # Stop after 5 errors
                        model_progress[model_name]["status"] = "Failed ❌"
                        print_status_table()
                        return
                    continue
                    
                df.at[idx, f'prediction_{model_name}'] = pred
                df.at[idx, f'reasoning_{model_name}'] = reason
                
                # Update progress (count total completed predictions)
                has_prediction = ~df[f'prediction_{model_name}'].isna()
                model_progress[model_name]["processed"] = sum(has_prediction)
                
                # Save after every prediction for Gemini
                if model_name == 'gemini':
                    df.to_csv(output_file, index=False)
                # Save every 2 predictions for other models
                elif model_progress[model_name]["processed"] % 2 == 0:
                    df.to_csv(output_file, index=False)
                
                # Update consensus prediction after each model update
                df['consensus_prediction'] = calculate_consensus_prediction(df)
                
                print_status_table()
                    
            except Exception as e:
                error_str = str(e)
                model_progress[model_name]["last_error"] = error_str
                error_count += 1
                if error_count >= 5:
                    model_progress[model_name]["status"] = "Failed ❌"
                    print_status_table()
                    return
                continue
            
            # Save progress after each prediction (for Gemini)
            if model_name == 'gemini':
                df.to_csv(output_file, index=False)
    
    model_progress[model_name]["status"] = "Complete ✓"
    print_status_table()

def calculate_consensus_prediction(df: pd.DataFrame) -> pd.Series:
    """Calculate consensus prediction based on majority vote from all models"""
    models = ['deepseek', 'gpt4', 'claude', 'gemini', 'perplexity', 'grok']
    
    def get_consensus(row):
        # Get all predictions for this row
        predictions = [row[f'prediction_{model}'] for model in models]
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
        if prediction_count > total_votes/2:
            return "Prediction"
        # If less than half vote for Prediction
        elif prediction_count < total_votes/2:
            return "Not Prediction"
        # If it's a tie (3/6), use Perplexity's vote
        else:
            perplexity_vote = row['prediction_perplexity']
            return perplexity_vote if pd.notna(perplexity_vote) else None

    return df.apply(get_consensus, axis=1)

async def classify_tweets_async(input_file: str, output_file: str):
    """Main function to classify tweets using multiple LLMs independently"""
    input_df = pd.read_csv(input_file)
    total_tweets = len(input_df)
    
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        df = input_df.copy()
        
    # Initialize consensus column if it doesn't exist
    if 'consensus_prediction' not in df.columns:
        df['consensus_prediction'] = None
    
    # Initialize columns if they don't exist
    models = {
        'deepseek': get_deepseek_prediction,
        'gpt4': get_gpt4_prediction,
        'claude': get_claude_prediction,
        'gemini': get_gemini_prediction,
        'perplexity': get_perplexity_prediction,
        'grok': get_grok_prediction
    }
    
    # Initialize progress for all models with total_tweets as denominator
    for model in models.keys():
        # Count how many predictions we already have
        has_prediction = ~df[f'prediction_{model}'].isna()
        completed = sum(has_prediction)
        
        model_progress[model] = {
            "processed": completed,
            "total": total_tweets,
            "status": "Complete ✓" if completed == total_tweets else "Waiting",
            "last_error": None
        }
    
    # Calculate initial consensus for existing predictions
    df['consensus_prediction'] = calculate_consensus_prediction(df)
    df.to_csv(output_file, index=False)
    
    print_status_table()
    
    # Create tasks for each model with their pending tweets
    tasks = []
    for model_name, model_func in models.items():
        # Find tweets that need this model's prediction
        needs_processing = df[f'prediction_{model_name}'].isna()
        tweets_to_process = [(idx, row['context']) 
                           for idx, row in df[needs_processing].iterrows()]
        
        if tweets_to_process:
            tasks.append(process_model(model_name, model_func, tweets_to_process, df, output_file))
    
    if tasks:
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print("\nError encountered during processing!")
            print_status_table()
            
            # Check if Gemini failed
            if model_progress['gemini']['status'] == "Failed ❌":
                print("\nGemini model failed. Would you like to:")
                print("1. Retry with adjusted safety settings")
                print("2. Skip Gemini and continue with other models")
                print("3. Stop processing entirely")
                
                choice = input("\nEnter your choice (1-3): ")
                if choice == "1":
                    print("\nRetrying Gemini with adjusted settings...")
                    # Reset Gemini progress
                    needs_processing = df['prediction_gemini'].isna()
                    tweets_to_process = [(idx, row['context']) 
                                       for idx, row in df[needs_processing].iterrows()]
                    model_progress['gemini'] = {
                        "processed": total_tweets - len(tweets_to_process),
                        "total": total_tweets,
                        "status": "Retrying",
                        "last_error": None
                    }
                    await process_model('gemini', get_gemini_prediction, tweets_to_process, df, output_file)
                elif choice == "2":
                    print("\nSkipping Gemini, continuing with other models...")
                    remaining_tasks = [t for t in tasks if t._coro.__name__ != 'process_model_gemini']
                    await asyncio.gather(*remaining_tasks)
                else:
                    print("\nStopping all processing...")
                    return
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    input_file = "/Users/davidlin/Desktop/results.csv"
    output_file = "/Users/davidlin/Desktop/classified_results.csv"
    asyncio.run(classify_tweets_async(input_file, output_file))
