# `DOCAL AI`

> This is a fork of original work done by [`David Lin`](https://github.com/davidlinjiahao) on [`frank-analyzer`](https://github.com/davidlinjiahao/frank_analyzer)

A Python-based tool wrapped in Javascript that analyses tweets from any public `@user` on X/Twitter using multiple Large Language Models (LLMs) and classifies them based on several abstract and user-provided metrics.

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
OPENROUTER_API_KEY=
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

```bash
# Shortcut to install python dependencies
npm run pip-install
```

### 3. Start proxy worker

```bash
# Shortcut to start the proxy worker
npm run proxy-dev
```

### 4. Start server

```bash
# Shortcut to start the server
npm run server-start
```

### 5. Start context creator

```bash
# Shortcut to start the context creator for a specific `@user`
npm run context $user
```

### 6. Start classifier

```bash
# Shortcut to start the classifier for a specific `@user`
npm run classify $user
```

### 7. Start extractor

```bash
# Shortcut to start the extractor for a specific `@user`
npm run extract $user
```

### 8. Start evaluator

```bash
# Shortcut to start the evaluator for a specific `@user`
npm run evaluate $user
```

### 9. Start scraper or indexer

#### 9.1. Start scraper for a tweet

```bash
# Shortcut to start the scraper for a specific `tweetId` with `flag`
npm run scrape $tweetId $flag
```

#### 9.2. Start indexer for a user

```bash
# Shortcut to start the indexer for a specific `@user` with `flag`
npm run index $user $flag
```

| Option         | Description                     | `flag`  |
| -------------- | ------------------------------- | ------- |
| Forced cookies | Use cookies from `.env` file    | `true`  |
| Cached cookies | Use previously cached cookies   | `false` |
| Fresh login    | Login without using any cookies | `none`  |
