import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

def test_grok():
    """Test Grok API with a simple request"""
    if not GROK_API_KEY:
        print("Error: GROK_API_KEY not found in environment variables")
        return False
        
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test message
    payload = {
        "model": "grok-2-latest",
        "messages": [
            {
                "role": "system",
                "content": "You are a test assistant."
            },
            {
                "role": "user",
                "content": "Testing. Just say hi and hello world and nothing else."
            }
        ],
        "stream": False,
        "temperature": 0
    }
    
    try:
        print("Testing Grok API connection...")
        print(f"API Key (first 5 chars): {GROK_API_KEY[:5]}...")
        print("Making request...")
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        print(f"Status code: {response.status_code}")
        print("Response headers:", response.headers)
        print("Response body:", response.text)
        
        response.raise_for_status()
        return True
        
    except Exception as e:
        print(f"Error testing Grok API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_grok()
    print(f"\nTest {'successful' if success else 'failed'}") 