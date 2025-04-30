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

## CUDA Build (optional)

To accelerate key derivation on NVIDIA GPUs, build the CUDA-based Ed25519 library:

### Linux
```bash
# Install CUDA Toolkit and build tools
# e.g. on Ubuntu:
sudo apt update && sudo apt install build-essential \
    nvidia-cuda-toolkit make
# Build shared library
make
```

### Windows
1. Install the NVIDIA CUDA Toolkit and Visual Studio with C++ support.
2. Open the "x64 Native Tools Command Prompt for VS".
3. Run:
```bat
cd %~dp0
nvcc -shared -o libed25519_cuda.dll gpu_ed25519.cu vendor\ed25519_ref10\ed25519_ref10.c \
    -I vendor\ed25519_ref10 -Xcompiler "/MD"
```
4. Copy `libed25519_cuda.dll` to the project root or ensure it’s in your `%PATH%`.

Then enable GPU mode in the UI (check "Use GPU acceleration") and restart the app.

## GitHub

Repository: https://github.com/DegenApeDev/solana-vanity-generator.git

## License

MIT License
