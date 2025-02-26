import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from pycoingecko import CoinGeckoAPI
import random

# Load arguments
user = sys.argv[1]

# Load environment variables
load_dotenv()


class CryptoConsensusAnalyser:
    def __init__(self):
        self.api_key = os.getenv("COINGECKO_API_KEY")
        if not self.api_key:
            raise ValueError("COINGECKO_API_KEY not found in environment variables")

        self.cg = CoinGeckoAPI()
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {"Accept": "application/json", "X-Cg-Api-Key": self.api_key}
        self.output_file = None
        self.error_count = 0

        # Set API key for CoinGeckoAPI instance
        self.cg.api_key = self.api_key

        # Token symbol to ID mapping
        self.symbol_to_id = {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "eth": "ethereum",
            "ethereum": "ethereum",
            "sol": "solana",
            "solana": "solana",
            "doge": "dogecoin",
            "dogecoin": "dogecoin",
            "xrp": "ripple",
            "ripple": "ripple",
            "ada": "cardano",
            "cardano": "cardano",
            "avax": "avalanche-2",
            "avalanche": "avalanche-2",
            "matic": "matic-network",
            "polygon": "matic-network",
            "link": "chainlink",
            "chainlink": "chainlink",
            "dot": "polkadot",
            "polkadot": "polkadot",
            "shib": "shiba-inu",
            "shibainu": "shiba-inu",
            "sacks": "sacks",
            "hawk": "hawk-tuah",
            "hawktuah": "hawk-tuah",
            "trump": "official-trump",
            "pepe": "pepe",
            "bonk": "bonk",
            "wif": "wif",
            "ai": "ai-network",
            "slop": "slop",
            "hype": "hyperliquid",
            "dust": "dust-protocol",
            "degod": "degod",
            "degods": "degod",
        }

        # Category mappings
        self.category_to_index = {
            "nfts": "non-fungible-tokens-nft",
            "nft": "non-fungible-tokens-nft",
            "memecoins": "meme-token",
            "memes": "meme-token",
            "meme": "meme-token",
            "rwa": "real-world-assets",
            "realworldassets": "real-world-assets",
            "agi": "ai-artificial-intelligence",
            "defi": "decentralised-finance-defi",
            "gaming": "gaming",
            "metaverse": "metaverse",
            "layer1": "layer-1",
            "l1": "layer-1",
            "layer2": "layer-2",
            "l2": "layer-2",
        }

        # Test API connection
        try:
            print("🔎 Testing API connection...")
            response = requests.get(f"{self.base_url}/ping", headers=self.headers)
            if response.status_code == 200:
                print(f"✅ API connection successful: {response.json()}")
            else:
                print(
                    f"❌ API connection failed: {response.status_code} - {response.text}"
                )
                raise Exception("Failed to connect to API")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to CoinGecko API: {str(e)}")
            raise

    def determine_target_type(self, target: str) -> Tuple[str, str]:
        """Determine if the target is a token or category and get its ID"""
        if not target or pd.isna(target):
            return "unknown", None

        # Clean the target string
        target = str(target).lower().strip()
        # Remove $ prefix if present
        target = target.lstrip("$")
        # Remove @ prefix if present
        target = target.lstrip("@")

        # First check symbol_to_id mapping
        if target in self.symbol_to_id:
            return "token", self.symbol_to_id[target]

        # Then check category mapping
        if target in self.category_to_index:
            return "category", self.category_to_index[target]

        # If not found in mappings, try to search CoinGecko
        try:
            print(f"🔎 Attempting to find match for unknown target: {target}")
            url = f"{self.base_url}/search"
            params = {"query": target}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                coins = data.get("coins", [])

                if coins:
                    # Get the first (most relevant) result
                    best_match = coins[0]
                    print(
                        f"✅ Found potential match: {best_match['id']} ({best_match['name']})"
                    )
                    return "token", best_match["id"]

        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")

        # If all attempts fail, treat as unknown
        return "unknown", target

    def get_current_price(self, target_id: str, target_type: str) -> Optional[float]:
        """Get current price for a token or category"""
        if target_type == "unknown" or not target_id:
            return None

        try:
            print(f"🔎 Getting current price for {target_type} {target_id}...")
            if target_type == "token":
                url = f"{self.base_url}/simple/price"
                params = {"ids": target_id, "vs_currencies": "usd"}
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    price = data.get(target_id, {}).get("usd")
                    if price is not None:
                        print(f"✅ Current price for {target_id}: {price}")
                        return price
            else:
                # For categories, use the markets endpoint
                url = f"{self.base_url}/coins/markets"
                params = {
                    "vs_currency": "usd",
                    "category": target_id,
                    "order": "market_cap_desc",
                    "per_page": 1,
                    "page": 1,
                    "sparkline": False,
                }
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        price = data[0].get("current_price")
                        if price is not None:
                            print(f"Current category price for {target_id}: {price}")
                            return price
            print(f"🔎 API Response: Status {response.status_code}")
            if response.status_code != 200:
                print(f"❌ Error response: {response.text}")
        except Exception as e:
            print(f"❌ Error getting current price: {str(e)}")
        return None

    def get_historical_price(
        self, target_id: str, target_type: str, date: datetime
    ) -> Optional[float]:
        """Get historical price for a token or category"""
        if target_type == "unknown" or not target_id:
            return None

        try:
            date_str = date.strftime("%d-%m-%Y")
            print(
                f"🔎 Getting historical price for {target_type} {target_id} at {date_str}..."
            )

            if target_type == "token":
                url = f"{self.base_url}/coins/{target_id}/history"
                params = {"date": date_str, "localization": "false"}
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    price = (
                        data.get("market_data", {}).get("current_price", {}).get("usd")
                    )
                    if price is not None:
                        print(f"✅ Historical price for {target_id}: {price}")
                        return price
            else:
                # For categories, use the markets endpoint
                url = f"{self.base_url}/coins/markets"
                params = {
                    "vs_currency": "usd",
                    "category": target_id,
                    "order": "market_cap_desc",
                    "per_page": 1,
                    "page": 1,
                    "sparkline": False,
                    "date": date_str,
                }
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        price = data[0].get("current_price")
                        if price is not None:
                            print(f"✅ Historical category price for {target_id}: {price}")
                            return price
            print(f"🔎 API Response: Status {response.status_code}")
            if response.status_code != 200:
                print(f"❌ Error response: {response.text}")
        except Exception as e:
            print(f"❌ Error getting historical price: {str(e)}")
        return None

    def save_results(self, results: List[Dict], final: bool = False):
        """Save results to CSV file"""
        if not self.output_file:
            return

        # Clean and format the results
        for result in results:
            # Clean context field
            if "context" in result and result["context"]:
                # Replace newlines and multiple spaces with single space
                result["context"] = " ".join(str(result["context"]).split())

        df = pd.DataFrame(results)

        # If this is the first save, write with headers
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            # Append without headers
            df.to_csv(self.output_file, mode="a", header=False, index=False)

    def process_predictions(
        self, predictions_df: pd.DataFrame, output_file: str
    ) -> pd.DataFrame:
        """Process predictions and calculate metrics"""
        print(f"\n🔎 Starting to process {len(predictions_df)} predictions...")
        self.output_file = output_file

        # Clear the output file if it exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            print(f"✅ Cleared existing output file: {self.output_file}")

        results = []
        total = len(predictions_df)
        processed = 0
        self.error_count = 0
        error_types = {"unknown_target": 0, "price_lookup": 0, "other": 0}

        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(predictions_df), batch_size):
            batch = predictions_df.iloc[i : i + batch_size]

            for _, row in batch.iterrows():
                processed += 1
                print(
                    f"\n🔎 Processing prediction {processed}/{total} ({(processed/total*100):.1f}%)"
                )
                print(f"➡️  Tweet ID: {row['tweet_id']}")
                print(f"➡️  Target: {row['consensus_target']}")
                print(f"➡️  Sentiment: {row['consensus_sentiment']}")

                error_occurred = False
                error_type = None

                try:
                    # Get target type and ID
                    target_type, target_token = self.determine_target_type(
                        row["consensus_target"]
                    )
                    print(
                        f"✅ Determined target type: {target_type}, token: {target_token}"
                    )

                    if target_type == "unknown":
                        error_occurred = True
                        error_type = "unknown_target"
                        error_types["unknown_target"] += 1
                        print(f"⚠️  Unknown target: {row['consensus_target']}")
                    else:
                        # Get timestamps
                        created_at = pd.to_datetime(row["createdAt"]).tz_localize(None)
                        current_time = datetime.now()
                        time_elapsed = (current_time - created_at).days

                        # Get prices
                        starting_price = self.get_historical_price(
                            target_token, target_type, created_at
                        )
                        ending_price = self.get_current_price(target_token, target_type)

                        # Check if we got valid prices
                        if starting_price is None or ending_price is None:
                            error_occurred = True
                            error_type = "price_lookup"
                            error_types["price_lookup"] += 1
                            print("❌ Failed to get valid prices")
                        else:
                            print(
                                f"✅ Got prices - Start: {starting_price:.4f}, End: {ending_price:.4f}"
                            )
                            # Calculate percentage change
                            pct_change = (
                                ((ending_price - starting_price) / starting_price) * 100
                                if starting_price != 0
                                else None
                            )
                            # Add result
                            results.append(
                                {
                                    "tweet_id": row["tweet_id"],
                                    "createdAt": row["createdAt"],
                                    "context": row["context"],
                                    "consensus_target": row["consensus_target"],
                                    "consensus_sentiment": row["consensus_sentiment"],
                                    "target_type": target_type,
                                    "target_token": target_token,
                                    "starting_price": starting_price,
                                    "ending_price": ending_price,
                                    "pct_change": pct_change,
                                    "time_elapsed": time_elapsed,
                                    "error_type": error_type,
                                }
                            )

                except Exception as e:
                    error_occurred = True
                    error_type = "other"
                    error_types["other"] += 1
                    print(f"❌ Error processing prediction: {str(e)}")

                # Increment error count if needed
                if error_occurred:
                    self.error_count += 1
                    results.append(
                        {
                            "tweet_id": row["tweet_id"],
                            "createdAt": row["createdAt"],
                            "context": row["context"],
                            "consensus_target": row["consensus_target"],
                            "consensus_sentiment": row["consensus_sentiment"],
                            "target_type": (
                                target_type if "target_type" in locals() else "error"
                            ),
                            "target_token": (
                                target_token if "target_token" in locals() else None
                            ),
                            "starting_price": None,
                            "ending_price": None,
                            "pct_change": None,
                            "time_elapsed": (
                                time_elapsed if "time_elapsed" in locals() else None
                            ),
                            "error_type": error_type,
                        }
                    )

            # Save batch results
            self.save_results(results[-len(batch) :])

            # Display progress
            if processed % batch_size == 0:
                self.display_progress(processed, total, error_types)

            # Sleep between batches to respect rate limits
            if i + batch_size < len(predictions_df):
                print("\n🔎 Waiting between batches...")
                time.sleep(1)

        # Display final stats
        print("\n➡️  Final Results:")
        self.display_progress(processed, total, error_types)

        return pd.DataFrame(results)

    def display_progress(self, processed, total, error_types):
        """Display progress and stats in a fixed position on the terminal"""
        progress = (processed / total * 100) if total > 0 else 0
        error_percentage = (self.error_count / processed * 100) if processed > 0 else 0

        # Print progress table
        progress_str = f"{processed}/{total}"
        print("\n=== Tweet Processing Status ===")
        print("┌─────────────┬───────────────┬───────────┬────────────┐")
        print("│ Metric      │ Count         │ Status    │ % of Total │")
        print("├─────────────┼───────────────┼───────────┼────────────┤")
        print(
            f"│ Progress    │ {progress_str:13} │ Running ⏯ │ {progress:>6.1f}%    │"
        )
        print("├─────────────┼───────────────┼───────────┼────────────┤")
        print(
            f"│ Errors      │ {self.error_count:<13,} │  Active ✓ │ {error_percentage:>6.1f}%    │"
        )
        print("└─────────────┴───────────────┴───────────┴────────────┘")

        # Print error breakdown
        if processed > 0:
            print("\n=== Error Breakdown ===")
            print("┌─────────────────┬───────────────┬─────────────────┐")
            print("│ Error Type      │ Count         │ % of Errors     │")
            print("├─────────────────┼───────────────┼─────────────────┤")
            for error_type, count in error_types.items():
                error_pct = (
                    (count / self.error_count * 100) if self.error_count > 0 else 0
                )
                print(
                    f"│ {error_type:<15} │ {count:<13,} │ {error_pct:>6.1f}%         │"
                )
            print("└─────────────────┴───────────────┴─────────────────┘")

        # Force output to display immediately
        sys.stdout.flush()


def main():
    input_file = os.path.expanduser(f"./results/{user}/extractor.csv")
    output_file = os.path.expanduser(f"./results/{user}/evaluator.csv")

    print("\n🔎 Initialising analyser...")
    analyser = CryptoConsensusAnalyser()

    try:
        # Read input file
        print("\n🔎 Reading input file...")
        df = pd.read_csv(input_file)
        print(f"✅ Read {len(df)} rows from {input_file}")

        # Process all predictions
        print("\n🔎 Starting prediction processing...")
        results_df = analyser.process_predictions(df, output_file)
        print("\n✅ Processing complete!")

        if os.path.exists(output_file):
            # Read and display summary of results
            final_results = pd.read_csv(output_file)
            print(f"\n✅ Processed {len(final_results)} rows")
            print("\n➡️  Columns in output file:")
            for col in final_results.columns:
                print(f"- {col}")

            # Display sample of first few rows
            print("\n➡️  Sample of first few rows:")
            print(final_results.head().to_string())

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
