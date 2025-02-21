# `OCSCO`: `On-Chain Social Credit Organiser`

> This is a fork of original work done by [`David Lin`](https://github.com/davidlinjiahao) on [`frank-analyzer`](https://github.com/davidlinjiahao/frank_analyzer)

A Python-based tool wrapped in Javascript that analyses tweets from any public `@` on X/Twitter using multiple Large Language Models (LLMs) and classifies them based on several abstract and user-provided metrics. 

## Features

- Multi-model analysis using:
  - `Claude 3.5 Sonnet`
  - `GPT-4o-mini`
  - `Gemini 1.5`
  - ~~`DeepSeek R1`~~
  - `Perplexity`
  - ~~`Grok 2`~~
  
- HTTP endpoints for web app integration
- Real-time consensus calculation
- Progress tracking with status table
- Automatic error handling and retries
- Asynchronous processing

## Setup

### 1. Clone the repository:
```bash
git clone https://github.com/docal-pro/docal-ai.git
cd docal-ai
```

### 2. Install dependencies:

### 2.1 `MacOS`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.2 `Linux` (`Ubuntu`):

On `Ubuntu-22.04` and above, you'll need to install `python3` virtual environment provider.

```bash
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### 3. Set up environment variables in `.env`:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
PERPLEXITY_API_KEY=
GROK_API_KEY=
DEEPSEEK_API_KEY=
```

## Usage

Run the script with:
```bash
python classifier.py
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
