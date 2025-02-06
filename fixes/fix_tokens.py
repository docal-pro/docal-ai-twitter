import pandas as pd
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('COINGECKO_API_KEY')

# API configuration
base_url = "https://pro-api.coingecko.com/api/v3"
headers = {
    'Accept': 'application/json',
    'X-Cg-Pro-Api-Key': api_key
}

# Token mapping fixes
token_fixes = {
    'trumpcoin': 'official-trump',
    'hawk-protocol': 'hawk-tuah',
    'slopfather': 'slop',
    'hyperliquid': 'hyperliquid',
    'unicorn-fart-dust': 'dust-protocol',
    'degod': 'degod'
}

# Specific tweet IDs that need fixing
specific_fixes = {
    '1859125558333886601': {'old_token': None, 'new_token': 'hyperliquid'},
    '1823659908300202433': {'old_token': None, 'new_token': 'hyperliquid'}
}

def get_historical_price(token_id: str, date: datetime, max_retries=3) -> float:
    for attempt in range(max_retries):
        try:
            date_str = date.strftime('%d-%m-%Y')
            url = f"{base_url}/coins/{token_id}/history"
            params = {
                'date': date_str,
                'localization': 'false'
            }
            print(f"Attempt {attempt + 1}/{max_retries} for historical price of {token_id} on {date_str}")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('market_data', {}).get('current_price', {}).get('usd')
                if price is not None:
                    return price
                print(f"No price data in response for {token_id} on {date_str}")
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit hit, waiting before retry...")
                time.sleep(5)
                continue
            else:
                print(f"API error {response.status_code}: {response.text}")
            
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait between retries
        except Exception as e:
            print(f"Error getting historical price for {token_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

def get_current_price(token_id: str, max_retries=3) -> float:
    for attempt in range(max_retries):
        try:
            url = f"{base_url}/simple/price"
            params = {
                'ids': token_id,
                'vs_currencies': 'usd'
            }
            print(f"Attempt {attempt + 1}/{max_retries} for current price of {token_id}")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get(token_id, {}).get('usd')
                if price is not None:
                    return price
                print(f"No price data in response for {token_id}")
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit hit, waiting before retry...")
                time.sleep(5)
                continue
            else:
                print(f"API error {response.status_code}: {response.text}")
            
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait between retries
        except Exception as e:
            print(f"Error getting current price for {token_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

# Read the CSV file
df = pd.read_csv(os.path.expanduser("~/Desktop/evaluated_results.csv"))

# First process specific tweet IDs
for tweet_id, fix_info in specific_fixes.items():
    mask = df['tweet_id'].astype(str) == tweet_id
    if any(mask):
        index = df[mask].index[0]
        new_token = fix_info['new_token']
        print(f"\nFixing specific tweet {tweet_id}: -> {new_token}")
        
        # Update token
        df.at[index, 'target_token'] = new_token
        df.at[index, 'target_type'] = 'token'  # Set the target type
        
        # Get new prices
        created_at = pd.to_datetime(df.at[index, 'createdAt']).tz_localize(None)
        starting_price = get_historical_price(new_token, created_at)
        ending_price = get_current_price(new_token)
        
        if starting_price is not None and ending_price is not None:
            df.at[index, 'starting_price'] = starting_price
            df.at[index, 'ending_price'] = ending_price
            df.at[index, 'pct_change'] = ((ending_price - starting_price) / starting_price) * 100 if starting_price != 0 else None
            df.at[index, 'error_type'] = None
            print(f"Updated prices - Start: {starting_price:.6f}, End: {ending_price:.6f}")
        else:
            df.at[index, 'error_type'] = 'price_lookup'
            print("Failed to get prices")

# Then process general token fixes
for index, row in df.iterrows():
    if row['target_token'] in token_fixes:
        old_token = row['target_token']
        new_token = token_fixes[old_token]
        print(f"\nUpdating row {index}: {old_token} -> {new_token}")
        
        # Update token
        df.at[index, 'target_token'] = new_token
        
        # Get new prices
        created_at = pd.to_datetime(row['createdAt']).tz_localize(None)
        starting_price = get_historical_price(new_token, created_at)
        ending_price = get_current_price(new_token)
        
        if starting_price is not None and ending_price is not None:
            df.at[index, 'starting_price'] = starting_price
            df.at[index, 'ending_price'] = ending_price
            df.at[index, 'pct_change'] = ((ending_price - starting_price) / starting_price) * 100 if starting_price != 0 else None
            df.at[index, 'error_type'] = None
            print(f"Updated prices - Start: {starting_price:.6f}, End: {ending_price:.6f}")
        else:
            df.at[index, 'error_type'] = 'price_lookup'
            print("Failed to get prices")

# Save the updated file
df.to_csv(os.path.expanduser("~/Desktop/evaluated_results.csv"), index=False)
print("\nUpdates complete!") 