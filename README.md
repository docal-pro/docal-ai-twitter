# Frank Analyzer

A Python-based tool that analyzes tweets from @frankdegods using multiple Language Models (LLMs) to identify market predictions.

## Features

- Multi-model analysis using:
  - Claude 3 Opus
  - GPT-4
  - Gemini Pro
  - DeepSeek
  - Perplexity
  - Grok
- Real-time consensus calculation
- Progress tracking with status table
- Automatic error handling and retries
- Asynchronous processing

## Setup

1. Clone the repository:
```bash
git clone https://github.com/davidlinjiahao/frank_analyzer.git
cd frank_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
GROK_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

## Usage

Run the script with:
```bash
python frank_classifier.py
```

The script will:
1. Process tweets using multiple LLMs
2. Calculate consensus predictions
3. Display real-time progress
4. Save results to CSV file

## Output

The script generates a CSV file with:
- Original tweet context
- Individual model predictions and reasoning
- Consensus prediction based on majority voting 