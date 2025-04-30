from flask import Flask, render_template, request, jsonify
from nacl.signing import SigningKey
import base58
try:
    import numpy as np
    import pycuda.autoinit  # initializes CUDA
    from pycuda import curandom
    rng = curandom.XORWOWRandomNumberGenerator()
except ImportError:
    rng = None
    import logging
    logging.warning("PyCUDA not available, GPU mode disabled")
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
    use_gpu = data.get('use_gpu', False)
    app.logger.info(f"GPU mode selected: {use_gpu}")
    start = time.time()
    result = {}
    found_event = threading.Event()
    lock = threading.Lock()

    def worker():
        local_tries = 0
        while not found_event.is_set():
            if use_gpu and rng:
                # generate 32-byte seed on GPU
                arr = rng.gen_uint32(8)
                seed = arr.astype(np.uint32).view(np.uint8).tobytes()
                sk = SigningKey(seed)
            else:
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
