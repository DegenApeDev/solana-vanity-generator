from flask import Flask, render_template, request, jsonify
from nacl.signing import SigningKey
import base58
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    start = time.time()
    tries = 0
    prefix_lower = prefix.lower()
    suffix_lower = suffix.lower()
    while True:
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        pub_bytes = verify_key.encode()
        pub = base58.b58encode(pub_bytes).decode()
        tries += 1
        pub_lower = pub.lower()
        if prefix_lower and suffix_lower:
            condition = pub_lower.startswith(prefix_lower) and pub_lower.endswith(suffix_lower)
        elif prefix_lower:
            condition = pub_lower.startswith(prefix_lower)
        elif suffix_lower:
            condition = pub_lower.endswith(suffix_lower)
        else:
            condition = True
        if condition:
            elapsed = time.time() - start
            secret_bytes = signing_key.encode() + pub_bytes
            secret = base58.b58encode(secret_bytes).decode()
            return jsonify(
                public_key=pub,
                secret_key=secret,
                tries=tries,
                elapsed=elapsed,
            )

if __name__ == '__main__':
    app.run(debug=True)
