import asyncio
import aiohttp
import pandas as pd
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
import openai
from PIL import Image
import io
import base64
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import sys
from collections import defaultdict

# Load arguments
user = sys.argv[1]

# Load environment variables
load_dotenv()

# Initialise API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Show all info messages
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/extractor.log"),  # Still log everything to file
        logging.StreamHandler(
            sys.stdout
        ),  # But only show important messages in console
    ],
)

# Constants
BACKUP_INTERVAL = 10  # Save backup every 10 processed items
CHECKPOINT_DIR = "checkpoints"
CACHE_DIR = "cache"
MAX_RETRIES = 3
RATE_LIMIT_PAUSE = 1  # seconds

# Token mappings for standardisation
TOKEN_MAPPINGS = {
    "BITCOIN": "BTC",
    "SACKS": "SACKS",  # Keep as is but ensure $ prefix
    "HAWK TUAH": "HAWKTUAH",  # Remove space
    # Add more token mappings as needed
}

# CoinGecko Category Mappings (id: display_name)
COINGECKO_CATEGORIES = {
    "artificial-intelligence": "AI",
    "non-fungible-tokens-nft": "NFTs",
    "decentralised-finance-defi": "DeFi",
    "layer-1": "L1",
    "layer-2": "L2",
    "meme-token": "Memes",
    "real-world-assets": "RWA",
    "social-money": "Social Fi",
    "play-to-earn": "GameFi",
    "bitcoin-ecosystem": "BTC",
    "fundraising": "ICO",
}

# Meta Categories aligned with CoinGecko
META_CATEGORIES = [
    "AI",  # Artificial Intelligence
    "NFTs",  # Non-Fungible Tokens
    "DeFi",  # Decentralised Finance
    "L1",  # Layer 1
    "L2",  # Layer 2
    "Memes",  # Meme Coins
    "RWA",  # Real World Assets
    "Social Fi",  # Social Finance
    "GameFi",  # Gaming Finance
    "BTC",  # Bitcoin Ecosystem
    "ICO",  # Initial Coin Offerings
]

# Project to token/category mapping
PROJECT_MAPPINGS = {
    "github": None,  # Not a crypto project
    "goatindex": None,  # Not found in CoinGecko
    # Add more project mappings as needed
}

SYSTEM_PROMPT = """You are analysing tweets from user @ to extract the target of his predictions and their sentiment. The target can be:

1. Specific tokens (tradable crypto tokens like $BTC, $ETH, $HYPE)
2. Projects (twitter accounts working in web3 without tokens yet)
3. Metas (crypto narratives aligned with CoinGecko categories: AI, NFTs, DeFi, L1, L2, Memes, RWA, Social Fi, GameFi, BTC, ICO)

For each prediction, identify ALL targets mentioned and whether user is bullish or bearish on them.
- For tokens, use standardised symbols (e.g., $BTC instead of $BITCOIN)
- For projects without tokens, return "None"
- For metas, use only the standardised categories listed above

Return your response in this exact JSON format:
{
    "targets": [
        {
            "name": "target name",
            "type": "token/project/meta",
            "sentiment": "bullish/bearish"
        }
    ],
    "reasoning": "brief explanation"
}"""

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Global progress tracking
model_progress = defaultdict(
    lambda: {"processed": 0, "total": 0, "status": "waiting", "text": "Waiting 𝌗", "last_error": None}
)


def initialise_model_progress(prediction_count: int, start_idx: int = 0):
    """Initialise progress tracking for all models"""
    global model_progress

    base_models = ["gpt4", "claude", "gemini", "perplexity", "grok", "deepseek"]
    model_progress.clear()  # Clear existing progress

    for model in base_models:
        model_progress[model] = {
            "processed": 0,  # Reset to 0
            "total": prediction_count,
            "status": "waiting",
            "text": "Waiting 𝌗",
            "last_error": None,
        }


def clear_terminal():
    """Clear the terminal screen and move cursor to top"""
    print(
        "\033[2J\033[H", end=""
    )  # ANSI escape codes to clear screen and move cursor to top
    sys.stdout.flush()


def print_status_table():
    """Print a clean status table of all models"""
    clear_terminal()
    print("\n=== Target Extraction Progress ===")
    print("┌────────────┬──────────────┬─────────────┐")
    print("│ Model      │ Progress     │ Status      │")
    print("├────────────┼──────────────┼─────────────┤")

    for model in sorted(model_progress.keys()):
        progress = model_progress[model]
        progress_str = f"{progress['processed']}/{progress['total']}"
        status = progress["status"]
        text = progress["text"]
        model_str = f"{model:10}"
        progress_str = f"{progress_str:12}"
        status_str = f"{status:11}"
        text_str = f"{text:11}"
        print(f"│ {model_str} │ {progress_str} │ {text_str} │")

    print("└────────────┴──────────────┴─────────────┘")
    sys.stdout.flush()  # Force flush output


class Cache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value
        if len(self.cache) % 10 == 0:  # Save every 10 new entries
            self.save_cache()


def retry_with_backoff(func, *args, max_retries=3, initial_delay=1):
    """Retry a function with exponential backoff"""
    for i in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if i == max_retries - 1:  # Last attempt
                raise e

            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                delay = initial_delay * (2**i)
            elif "401" in error_str or "unauthorized" in error_str:
                raise e
            elif "quota" in error_str or "credit" in error_str:
                raise e
            else:
                delay = initial_delay

            time.sleep(delay)
            continue


async def get_deepseek_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using DeepSeek"""

    def _make_request():
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise Exception("401 Unauthorised - DeepSeek API key not found")

        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}",
                },
            ],
            "stream": False,
        }
        response = requests.post(
            "https://api.deepseek.com/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        result = response.json()

        if not result or "choices" not in result or not result["choices"]:
            raise Exception("Empty or invalid response from DeepSeek")

        return json.loads(result["choices"][0]["message"]["content"])

    return await asyncio.to_thread(retry_with_backoff, _make_request)


async def get_gpt4_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using GPT-4"""
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}",
                },
            ],
            temperature=0.0,
            max_tokens=150,
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse GPT-4 response for text: {text}")
        return {"targets": [], "reasoning": "Failed to parse response"}
    except Exception as e:
        logging.error(f"GPT-4 API error: {str(e)}")
        raise


async def get_claude_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using Claude Opus"""
    try:
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
            model="claude-3-opus-20240229",
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}",
                }
            ],
            temperature=0.0,
        )
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse Claude response for text: {text}")
        return {"targets": [], "reasoning": "Failed to parse response"}
    except Exception as e:
        logging.error(f"Claude API error: {str(e)}")
        raise


async def get_gemini_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using Google Gemini"""
    try:
        model = genai.GenerativeModel("gemini-pro")

        generation_config = genai.GenerationConfig(
            temperature=0.0, candidate_count=1, max_output_tokens=150
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

        prompt = f"""TASK: Extract prediction targets from this tweet.
        
Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}

{SYSTEM_PROMPT}"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response text
            text = response.text
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            raise

    except json.JSONDecodeError:
        logging.error(f"Failed to parse Gemini response for text: {text}")
        return {"targets": [], "reasoning": "Failed to parse response"}
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        raise


async def get_perplexity_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using Perplexity"""
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "pplx-7b-online",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 400:
                    error_text = await response.text()
                    logging.error(f"Perplexity API 400 error: {error_text}")
                    return {"targets": [], "reasoning": "API error"}
                response.raise_for_status()
                result = await response.json()
                try:
                    return json.loads(result["choices"][0]["message"]["content"])
                except (KeyError, json.JSONDecodeError) as e:
                    logging.error(f"Failed to parse Perplexity response: {str(e)}")
                    return {"targets": [], "reasoning": "Failed to parse response"}
    except aiohttp.ClientError as e:
        logging.error(f"Perplexity API connection error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Perplexity API error: {str(e)}")
        raise


async def get_grok_prediction(text: str, timestamp: datetime) -> Dict:
    """Get prediction target using Grok"""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Tweet from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {text}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:  # Reduced timeout
            async with session.post(
                "https://api.x.ai/v1/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status == 429:  # Rate limit hit
                    await asyncio.sleep(10)  # Reduced wait time
                    return {"targets": [], "reasoning": "Rate limit hit"}

                if response.status == 400:
                    return {"targets": [], "reasoning": "API error"}

                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]

                # Remove code block markers if present
                if content.startswith("```json\n"):
                    content = content[8:]
                if content.endswith("\n```"):
                    content = content[:-4]

                try:
                    return json.loads(content)
                except (KeyError, json.JSONDecodeError):
                    return {"targets": [], "reasoning": "Failed to parse response"}

    except asyncio.TimeoutError:
        return {"targets": [], "reasoning": "Timeout"}
    except aiohttp.ClientError:
        return {"targets": [], "reasoning": "Connection error"}
    except Exception:
        return {"targets": [], "reasoning": "API error"}


async def get_category_market_data(category_id: str) -> Dict:
    """Get market data for a specific CoinGecko category"""
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "category": category_id,
        "order": "market_cap_desc",
        "per_page": 1,  # We just need category totals
        "page": 1,
        "sparkline": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 429:  # Rate limit
                    await asyncio.sleep(60)  # Wait longer for rate limits
                    return None

                data = await response.json()
                if not data:
                    return None

                # Get category market data
                return {
                    "market_cap": sum(
                        coin["market_cap"] for coin in data if coin.get("market_cap")
                    ),
                    "volume_24h": sum(
                        coin["total_volume"]
                        for coin in data
                        if coin.get("total_volume")
                    ),
                    "price_change_24h": (
                        data[0].get("price_change_percentage_24h", 0) if data else 0
                    ),
                }
    except Exception as e:
        logging.error(f"Error getting category market data: {str(e)}")
        return None


async def process_meta_target(target_name: str) -> Tuple[str, Dict]:
    """Process a meta target and get its market data"""
    # Find matching CoinGecko category
    category_id = None
    display_name = None

    for cat_id, name in COINGECKO_CATEGORIES.items():
        if name.upper() == target_name.upper():
            category_id = cat_id
            display_name = name
            break

    if not category_id:
        return None, None

    # Get market data for the category
    market_data = await get_category_market_data(category_id)

    return display_name, market_data


class TargetExtractor:
    def __init__(self):
        self.cache = Cache(os.path.join(CACHE_DIR, "target_cache.json"))
        self.session = None
        self.processed_count = 0
        self.last_backup = 0
        self.checkpoint_file = os.path.join(
            CHECKPOINT_DIR, "processing_checkpoint.json"
        )

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def extract_target(self, text: str, timestamp: datetime) -> Dict:
        """Extract prediction target using multiple LLM models in parallel"""
        models = {"gemini": get_gemini_prediction, "grok": get_grok_prediction}

        # Create tasks for all models and run them concurrently
        tasks = []
        for model_name, model_func in models.items():
            if model_progress[model_name]["status"] not in [
                "failed"
            ]:  # Only exclude permanently failed models
                task = asyncio.create_task(
                    self._run_model(model_name, model_func, text, timestamp)
                )
                tasks.append(task)

        # Wait for all tasks to complete or timeout individually
        completed_tasks = []
        try:
            # Use asyncio.as_completed to process results as they come in
            for future in asyncio.as_completed(tasks, timeout=30):
                try:
                    await future
                    completed_tasks.append(future)
                except asyncio.TimeoutError:
                    continue  # Skip timed out task but continue with others
                except Exception as e:
                    logging.error(f"Task error: {str(e)}")
                    continue
        except asyncio.TimeoutError:
            logging.warning("Some models timed out")

        # Collect results from successful model runs
        results = {}
        for model_name in models.keys():
            if model_progress[model_name].get("last_result") is not None:
                results[model_name] = model_progress[model_name]["last_result"]
                del model_progress[model_name]["last_result"]  # Clean up

        # Process results for each model
        processed_results = []
        for model_name, result in results.items():
            if not isinstance(result, dict) or "targets" not in result:
                continue

            for target in result.get("targets", []):
                try:
                    if (
                        not isinstance(target, dict)
                        or "type" not in target
                        or "name" not in target
                    ):
                        continue

                    formatted_name = target["name"]
                    market_data = None

                    if target["type"] == "token":
                        # Standardise token names
                        token_name = formatted_name.upper().replace("$", "")
                        token_name = TOKEN_MAPPINGS.get(token_name, token_name)
                        formatted_name = f"${token_name}"

                    elif target["type"] == "project":
                        # Check project mappings
                        project_name = formatted_name.lower().replace("@", "")
                        if project_name in PROJECT_MAPPINGS:
                            if PROJECT_MAPPINGS[project_name] is None:
                                formatted_name = "None"
                            else:
                                formatted_name = PROJECT_MAPPINGS[project_name]
                        else:
                            formatted_name = f"@{project_name}"

                    elif target["type"] == "meta":
                        # Map to CoinGecko category and get market data
                        formatted_name, market_data = await process_meta_target(
                            formatted_name
                        )
                        if not formatted_name:
                            continue

                    if formatted_name != "None":
                        result_entry = {
                            "model": model_name,
                            "target": formatted_name,
                            "type": target["type"],
                            "sentiment": target.get("sentiment", "unknown"),
                            "reasoning": result.get("reasoning", ""),
                        }

                        # Add market data if available
                        if market_data:
                            result_entry.update(
                                {
                                    "market_cap": market_data["market_cap"],
                                    "volume_24h": market_data["volume_24h"],
                                    "price_change_24h": market_data["price_change_24h"],
                                }
                            )

                        processed_results.append(result_entry)

                except Exception as e:
                    logging.error(
                        f"Error processing target from {model_name}: {str(e)}"
                    )
                    continue

        return {
            "targets": processed_results,
            "reasoning": " | ".join(
                r.get("reasoning", "") for r in results.values() if isinstance(r, dict)
            ),
        }

    async def _run_model(
        self, model_name: str, model_func, text: str, timestamp: datetime
    ):
        """Run a single model and handle its results/errors"""
        try:
            model_progress[model_name]["status"] = "running"
            model_progress[model_name]["text"] = "Running ⏯"
            print_status_table()

            # Execute model with timeout
            try:
                result = await asyncio.wait_for(
                    model_func(text, timestamp), timeout=20
                )  # Shorter individual timeout

                if not isinstance(result, dict) or "targets" not in result:
                    model_progress[model_name]["status"] = "error"
                    model_progress[model_name]["text"] = "  Error ✕"
                    print_status_table()
                    return

                # Store result and update progress
                model_progress[model_name]["last_result"] = result
                model_progress[model_name]["processed"] += 1

                # Update status based on result
                if "rate limit" in result.get("reasoning", "").lower():
                    model_progress[model_name]["status"] = "quota"
                    model_progress[model_name]["text"] = "  Quota ✕"
                else:
                    model_progress[model_name]["status"] = "running"
                    model_progress[model_name]["text"] = "Running ⏯"

                # Save progress after each successful model extraction
                self._save_model_progress(
                    model_name, model_progress[model_name]["processed"]
                )
                print_status_table()

            except asyncio.TimeoutError:
                model_progress[model_name]["status"] = "timeout"
                model_progress[model_name]["text"] = "Timeout ✕"
                print_status_table()
                return

        except Exception as e:
            error_str = str(e)
            if "quota" in error_str.lower() or "credit" in error_str.lower():
                model_progress[model_name]["status"] = "failed"
                model_progress[model_name]["text"] = " Failed ✕"
            else:
                model_progress[model_name]["status"] = "error"
                model_progress[model_name]["text"] = "  Error ✕"
            print_status_table()

    def _save_model_progress(self, model_name: str, processed_count: int):
        """Save individual model progress to a file"""
        progress_file = os.path.join(CHECKPOINT_DIR, f"{model_name}_progress.json")
        with open(progress_file, "w") as f:
            json.dump(
                {
                    "processed": processed_count,
                    "total": model_progress[model_name]["total"],
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )

    def _load_model_progress(self, model_name: str) -> Optional[int]:
        """Load saved progress for a model"""
        progress_file = os.path.join(CHECKPOINT_DIR, f"{model_name}_progress.json")
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
                return data["processed"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _save_checkpoint(self, last_processed_idx: int):
        """Save checkpoint of last processed index"""
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {
                    "last_processed_idx": last_processed_idx,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )

    def _load_checkpoint(self) -> Optional[int]:
        """Load last processed index from checkpoint"""
        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                return data["last_processed_idx"]
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _calculate_consensus(
        self, row_data: Dict, base_models: List[str]
    ) -> Tuple[str, str]:
        """Calculate consensus target and sentiment from all model predictions"""
        valid_targets = []
        valid_sentiments = []
        no_target_count = 0

        for model in base_models:
            target = row_data.get(f"{model}_target")
            sentiment = row_data.get(f"{model}_sentiment")

            # Count explicit "No target found" responses
            if target == "No target found":
                no_target_count += 1
                continue

            # Only include valid predictions
            if target and target not in [
                "Error: failed",
                "Error: error",
                "Error: timeout",
            ]:
                valid_targets.append(target)
                if sentiment and sentiment not in ["No sentiment", "unknown"]:
                    valid_sentiments.append(sentiment)

        # Calculate consensus target
        if len(valid_targets) == 0:
            # If majority of non-error responses are "No target found", use that
            valid_responses = len(
                [
                    t
                    for t in [row_data.get(f"{m}_target") for m in base_models]
                    if t and not t.startswith("Error:")
                ]
            )
            if valid_responses > 0 and no_target_count / valid_responses > 0.5:
                consensus_target = "No target found"
            else:
                consensus_target = "No consensus"
        else:
            target_counts = {}
            for target in valid_targets:
                target_counts[target] = target_counts.get(target, 0) + 1
            consensus_target = max(target_counts.items(), key=lambda x: x[1])[0]

        # Calculate consensus sentiment
        if len(valid_sentiments) == 0 or consensus_target == "No target found":
            consensus_sentiment = "No sentiment"
        else:
            sentiment_counts = {}
            for sentiment in valid_sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            consensus_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        return consensus_target, consensus_sentiment

    async def process_file(self, input_file: str, output_file: str):
        try:
            await self.init_session()

            # Read and validate input file
            print("🔎 Reading input file...")
            df = pd.read_csv(input_file)

            # Filter for predictions only
            df = df[df["consensus_prediction"] == "Prediction"].copy()
            df["createdAt"] = pd.to_datetime(df["createdAt"])
            total_predictions = len(df)
            print(f"🔎 Processing {total_predictions} predictions")

            # Load checkpoint if exists
            start_idx = self._load_checkpoint() or 0
            if start_idx > 0:
                print(f"🔎 Resuming from checkpoint at index {start_idx}")
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file)
                    if len(existing_df) != start_idx:
                        print("⚠️  Warning: Checkpoint mismatch. Starting from beginning.")
                        start_idx = 0

            # Initialise progress for Gemini and Grok only
            model_progress.clear()

            base_models = ["gemini", "grok"]
            for model in base_models:
                model_progress[model] = {
                    "processed": 0,
                    "total": total_predictions,
                    "status": "waiting",
                    "text": "Waiting 𝌗",
                    "last_error": None,
                }
            print_status_table()

            # Initialise output DataFrame with correct column structure
            column_order = [
                "tweet_id",
                "createdAt",
                "context",
                "consensus_prediction",
                "consensus_target",
                "consensus_sentiment",
            ]

            # Add model-specific columns for Gemini and Grok only
            for model in base_models:
                column_order.extend(
                    [f"{model}_target", f"{model}_sentiment", f"{model}_reasoning"]
                )

            # Create new output file if starting from beginning
            if start_idx == 0:
                output_df = pd.DataFrame(columns=column_order)
                output_df.to_csv(output_file, index=False)
                print(
                    f"✅ Created new output file with columns: {', '.join(column_order)}"
                )

            # Process each prediction sequentially
            for idx, row in df.iloc[start_idx:].iterrows():
                try:
                    print(f"\n🔎 Processing tweet {idx + 1}/{total_predictions}:")
                    print(f"➡️  Tweet ID: {row['tweet_id']}")
                    print(f"➡️  Created At: {row['createdAt']}")
                    print(f"➡️  Context: {row['context'][:100]}...")

                    result = await self.extract_target(row["context"], row["createdAt"])

                    # Process results for this prediction
                    base_row = {
                        "tweet_id": row["tweet_id"],
                        "createdAt": row["createdAt"],
                        "context": row["context"],
                        "consensus_prediction": "Prediction",
                    }

                    # Process Gemini and Grok results only
                    row_data = base_row.copy()
                    for model in base_models:
                        model_result = next(
                            (
                                t
                                for t in result.get("targets", [])
                                if t["model"] == model
                            ),
                            None,
                        )

                        if model_result and model_result.get("target"):
                            row_data[f"{model}_target"] = model_result["target"]
                            row_data[f"{model}_sentiment"] = model_result.get(
                                "sentiment", "unknown"
                            )
                            row_data[f"{model}_reasoning"] = model_result.get(
                                "reasoning", ""
                            )
                        else:
                            status = model_progress[model]["status"]
                            if status in ["failed", "error", "timeout"]:
                                error_msg = f"Error: {status}"
                            else:
                                error_msg = "No target found"

                            row_data[f"{model}_target"] = error_msg
                            row_data[f"{model}_sentiment"] = error_msg
                            row_data[f"{model}_reasoning"] = error_msg

                    # Calculate consensus using only Gemini and Grok
                    consensus_target, consensus_sentiment = self._calculate_consensus(
                        row_data, base_models
                    )
                    row_data["consensus_target"] = consensus_target
                    row_data["consensus_sentiment"] = consensus_sentiment

                    # Ensure all columns are present and in correct order
                    batch_df = pd.DataFrame(
                        [{col: row_data.get(col, "") for col in column_order}]
                    )
                    batch_df.to_csv(output_file, mode="a", header=False, index=False)

                    # Save checkpoint
                    self._save_checkpoint(idx + 1)

                    print(f"✅ Saved results for tweet {idx + 1}")
                    print(
                        f"Consensus - Target: {consensus_target}, Sentiment: {consensus_sentiment}"
                    )

                    print_status_table()

                except Exception as e:
                    print(f"❌ Error processing row {idx}: {str(e)}")
                    continue

            print("\n✅ Processing complete!")
            print_status_table()

        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            raise

        finally:
            await self.close_session()

    def _save_backup(self, results: List[Dict]):
        """Save backup of processed results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(CHECKPOINT_DIR, f"backup_{timestamp}.json")
        with open(backup_file, "w") as f:
            json.dump(results, f)
        logging.info(f"✅ Backup saved: {backup_file}")


async def check_files():
    print("\n🔎 Checking files and configuration...")
    input_file = f"./results/{user}/classifier.csv"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False

    print(f"✅ Input file found: {input_file}")

    try:
        print("🔎 Reading input file...")
        df = pd.read_csv(input_file)
        print(f"➡️  Total rows in input file: {len(df)}")

        # Ensure required columns exist
        required_columns = ["tweet_id", "createdAt", "context", "consensus_prediction"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        # Filter for actual predictions using consensus_prediction column
        print("🔎 Filtering predictions...")
        predictions_df = df[df["consensus_prediction"] == "Prediction"].copy()
        prediction_count = len(predictions_df)

        print(
            f"✅ Found {prediction_count} predictions to process out of {len(df)} total rows"
        )

        if prediction_count == 0:
            print("❌ No predictions found to process. Exiting...")
            return False

        print("🔎 Converting timestamps...")
        predictions_df["createdAt"] = pd.to_datetime(predictions_df["createdAt"])

        # Sort by createdAt to ensure chronological processing
        predictions_df = predictions_df.sort_values("createdAt")

        # Save filtered predictions
        cleaned_file = f"./results/{user}/cleaned.csv"
        print(f"🔎 Saving filtered predictions to: {cleaned_file}")
        predictions_df.to_csv(cleaned_file, index=False)

        # Initialise progress tracking with fresh counters - only for Gemini and Grok
        model_progress.clear()  # Clear existing progress

        for model in ["gemini", "grok"]:
            model_progress[model] = {
                "processed": 0,
                "total": prediction_count,
                "status": "waiting",
                "text": "Waiting 𝌗",
                "last_error": None,
            }
        print_status_table()

        return True

    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        return False


async def main():
    print("\n🔎 Starting extraction process...")

    if not await check_files():
        print("❌ Initialisation checks failed. Exiting...")
        return

    try:
        print("\n🔎 Initialising extractor...")
        extractor = TargetExtractor()
        input_file = f"./results/{user}/classifier.csv"
        output_file = f"./results/{user}/extractor.csv"
        print(f"\n🔎 Processing file: {input_file}")
        print(f"🔎 Output will be saved to: {output_file}")
        await extractor.process_file(input_file, output_file)
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("❌ Process interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {str(e)}", exc_info=True)
