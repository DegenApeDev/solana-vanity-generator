# Solana Vanity Wallet Generator

A simple Flask web app to generate Solana vanity wallets (Ed25519 keypairs) with a custom prefix. Uses PyNaCl for key generation and Base58 encoding to match Solana format.

## Features

- Specify a desired prefix for your Solana public key
- Displays public key, secret key, number of attempts, and elapsed time
- Responsive Bootstrap-based UI

## Prerequisites

- Python 3.7+
- pip

## Installation

```bash
python3 -m venv venv
source venv/bin/activate      # or `source venv/bin/activate.fish` for fish shell
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, enter a prefix, and click **Generate**.

## GitHub

Repository: https://github.com/DegenApeDev/solana-vanity-generator.git

## License

MIT License
