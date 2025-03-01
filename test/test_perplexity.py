import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def test_perplexity():
    """Test Perplexity API with a simple request"""
    if not PERPLEXITY_API_KEY:
        print("Error: PERPLEXITY_API_KEY not found in environment variables")
        return False
        
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test message
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "user",
                "content": "Say hello"
            }
        ]
    }
    
    try:
        print("Testing Perplexity API connection...")
        print(f"API Key (first 5 chars): {PERPLEXITY_API_KEY[:5]}...")
        print("Headers:", headers)
        print("Payload:", payload)
        print("Making request...")
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers=headers
        )
        
        print(f"Status code: {response.status_code}")
        print("Response headers:", dict(response.headers))
        print("Response body:", response.text)
        
        response.raise_for_status()
        return True
        
    except Exception as e:
        print(f"Error testing Perplexity API: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_perplexity()
    print(f"\nTest {'successful' if success else 'failed'}") 