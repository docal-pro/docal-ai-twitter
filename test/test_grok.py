import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


async def test_grok():
    """Test Grok API with a simple prediction"""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    test_text = "BTC to 100k"
    
    payload = {
        "model": "grok-2",
        "messages": [
            {
                "role": "system",
                "content": """You are analyzing tweets to extract prediction targets and sentiment.
For each prediction, identify the target and whether it's bullish or bearish.
Return in JSON format:
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
            },
            {"role": "user", "content": f"Tweet: {test_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    print("Testing Grok API...")
    print(f"API Key (first 5 chars): {os.getenv('GROK_API_KEY')[:5]}...")
    print(f"Test text: {test_text}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers) as response:
                print(f"Response status: {response.status}")
                response_text = await response.text()
                print(f"Raw response: {response_text}")
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Remove code block markers if present
                    if content.startswith('```json\n'):
                        content = content[8:]  # Remove ```json\n
                    if content.endswith('\n```'):
                        content = content[:-4]  # Remove \n```
                    
                    try:
                        parsed_content = json.loads(content)
                        print("\nParsed response:")
                        print(json.dumps(parsed_content, indent=2))
                        return True
                    except Exception as e:
                        print(f"Error parsing response: {str(e)}")
                        return False
                else:
                    print(f"Error: {response.status}")
                    return False
                
    except Exception as e:
        print(f"Error connecting to Grok API: {str(e)}")
        return False


async def main():
    success = await test_grok()
    if success:
        print("\nGrok API test successful! ✓")
    else:
        print("\nGrok API test failed! ✗")


if __name__ == "__main__":
    asyncio.run(main()) 
