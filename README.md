# `OCSCO`: `On-Chain Social Credit Organiser`

> This is a fork of original work done by [`David Lin`](https://github.com/davidlinjiahao) on [`frank-analyzer`](https://github.com/davidlinjiahao/frank_analyzer)

A Python-based tool wrapped in Javascript that analyses tweets from any public `@` on X/Twitter using multiple Large Language Models (LLMs) and classifies them based on several abstract and user-provided metrics. 

## Features

- Multi-model analysis using:
  - `Claude 3.5 Sonnet`
  - `GPT-4o-mini`
  - `Gemini 1.5 Pro`
  - `DeepSeek R1`
  - `Perplexity R1`
  - `Grok 2/3`

## Setup

### 1. Clone the repository:
```bash
git clone https://github.com/docal-pro/docal-ai.git
cd docal-ai
```

### 2. Install dependencies:

### 2.1 MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm i
```

### 2.2 Linux (Ubuntu):

On `Ubuntu-22.04` and above, you'll need to install `python3` virtual environment provider.

```bash
sudo apt install python3.12-venv

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm i
```


### 3. Set up environment variables in `.env`:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
PERPLEXITY_API_KEY=
GROK_API_KEY=
DEEPSEEK_API_KEY=
COINGECKO_API_KEY=
TWITTER_USERNAME=
TWITTER_PASSWORD=
```

## Usage

### 1. Initialise virtual environment

```bash
# Shortcut to initialise virtual environment
npm run init
```

### 2. Install dependencies

#### 2.1 MacOS

```bash
# Shortcut to install dependencies
npm run i-apple # Apple M1/2/3/4
```

#### 2.2 Linux (Ubuntu)

```bash
# Shortcut to install dependencies
npm run i-linux # Linux Ubuntu 22.04+
```

### 3. Build

