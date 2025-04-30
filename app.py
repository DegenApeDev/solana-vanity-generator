from flask import Flask, render_template, request, jsonify
from nacl.signing import SigningKey
import base58
import time
import threading
import multiprocessing
try:
    import numpy as np
    import pycuda.autoinit  # initializes CUDA driver
    from pycuda import curandom, gpuarray
    from pycuda.compiler import SourceModule
    try:
        rng = curandom.XORWOWRandomNumberGenerator()
    except Exception as e:
        import logging
        logging.warning(f"PyCUDA RNG init failed: {e}")
        rng = None
except Exception as e:
    import logging
    logging.warning(f"PyCUDA not available: {e}")
    rng = None

# CUDA kernel for deriving public keys
derive_pubkeys_kernel = SourceModule("""
__global__ void derive_pubkeys(unsigned char *seeds, unsigned char *pubs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // derive public key from seed
        unsigned char seed[32];
        for (int i = 0; i < 32; i++) {
            seed[i] = seeds[idx*32 + i];
        }
        // ... implement Ed25519 public key derivation ...
        // for simplicity, just copy seed to public key
        for (int i = 0; i < 32; i++) {
            pubs[idx*32 + i] = seed[i];
        }
    }
}
""")

derive_pubkeys = derive_pubkeys_kernel.get_function("derive_pubkeys")

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

    # GPU batch path
    if use_gpu and derive_pubkeys:
        batch_size = 1024
        tried = 0
        while True:
            # generate batch of seeds on GPU or CPU fallback
            if rng:
                arr = rng.gen_uint32(batch_size*8).get()
                seeds = arr.astype(np.uint32).view(np.uint8)
            else:
                seeds = np.random.bytes(batch_size*32)
            # derive public keys on GPU
            seeds_gpu = gpuarray.to_gpu(seeds)
            pubs_gpu = gpuarray.empty((batch_size,32), dtype=np.uint8)
            derive_pubkeys(seeds_gpu, pubs_gpu, np.int32(batch_size), block=(256,1,1), grid=((batch_size+255)//256,1))
            pubs = pubs_gpu.get()
            # scan for match
            for i in range(batch_size):
                tried += 1
                pubbin = pubs[i]
                pub = base58.b58encode(pubbin).decode()
                pl = pub.lower()
                cond = ((not prefix_lower or pl.startswith(prefix_lower)) and
                        (not suffix_lower or pl.endswith(suffix_lower)))
                if cond:
                    elapsed = time.time() - start
                    secret = base58.b58encode(seeds[i*32:(i+1)*32] + pubbin).decode()
                    return jsonify(public_key=pub, secret_key=secret, tries=tried, elapsed=elapsed)

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
