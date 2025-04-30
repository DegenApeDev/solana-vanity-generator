from flask import Flask, render_template, request, jsonify
from nacl.signing import SigningKey
import base58
import time
import threading
import multiprocessing

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    prefix_lower = prefix.lower()
    suffix_lower = suffix.lower()
    start = time.time()
    result = {}
    found_event = threading.Event()
    lock = threading.Lock()

    def worker():
        local_tries = 0
        while not found_event.is_set():
            sk = SigningKey.generate()
            vk = sk.verify_key
            pub = base58.b58encode(vk.encode()).decode()
            local_tries += 1
            pl = pub.lower()
            if prefix_lower and suffix_lower:
                cond = pl.startswith(prefix_lower) and pl.endswith(suffix_lower)
            elif prefix_lower:
                cond = pl.startswith(prefix_lower)
            elif suffix_lower:
                cond = pl.endswith(suffix_lower)
            else:
                cond = True
            if cond:
                elapsed = time.time() - start
                secret = base58.b58encode(sk.encode() + vk.encode()).decode()
                with lock:
                    if not result:
                        result['public_key'] = pub
                        result['secret_key'] = secret
                        result['tries'] = local_tries
                        result['elapsed'] = elapsed
                found_event.set()
                break

    num_threads = multiprocessing.cpu_count()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    found_event.wait()
    for t in threads:
        t.join()

    return jsonify(**result)

if __name__ == '__main__':
    app.run(debug=True)
