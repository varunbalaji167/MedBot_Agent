"""
Standalone diagnostic -- run this directly, NOT through Streamlit:

    python3 diagnose_env.py

It imports and exercises each heavy library one step at a time, printing
after each one succeeds. Since native crashes (segfaults) bypass Python's
exception handling entirely, there's no traceback to read -- the only signal
we get is "which printed line was the last one before the process died."
Whatever step's "OK" line does NOT print is the actual culprit.

Paste back everything this prints, plus the very last line the terminal
shows (segfault / abort / etc.), and that pinpoints the real cause instead
of guessing further.
"""

import sys
import platform

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Machine/arch:", platform.machine())
print("Platform:", platform.platform())
print(flush=True)

print("Step 1: importing numpy...", flush=True)
import numpy as np
print("  OK, numpy", np.__version__, flush=True)

print("Step 2: importing torch...", flush=True)
import torch
print("  OK, torch", torch.__version__, flush=True)
print("  torch backends -- mps available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else "n/a", flush=True)

print("Step 3: importing sentence_transformers...", flush=True)
from sentence_transformers import SentenceTransformer
print("  OK", flush=True)

print("Step 4: loading MiniLM model (downloads on first run)...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  OK, model loaded", flush=True)

print("Step 5: encoding a test sentence with MiniLM...", flush=True)
vec = model.encode(["hello world"])
print("  OK, vector shape:", vec.shape, flush=True)

print("Step 6: importing faiss...", flush=True)
import faiss
print("  OK, faiss module:", faiss.__file__, flush=True)

print("Step 7: setting faiss to single-threaded (faiss.omp_set_num_threads(1))...", flush=True)
faiss.omp_set_num_threads(1)
print("  OK", flush=True)

print("Step 8: building a tiny faiss index (IndexIDMap2 over IndexFlatIP)...", flush=True)
index = faiss.IndexIDMap2(faiss.IndexFlatIP(vec.shape[1]))
print("  OK, index created", flush=True)

print("Step 9: adding one vector to the index...", flush=True)
ids = np.array([0], dtype="int64")
index.add_with_ids(vec.astype("float32"), ids)
print("  OK, ntotal =", index.ntotal, flush=True)

print("Step 10: searching the index...", flush=True)
scores, idxs = index.search(vec.astype("float32"), 1)
print("  OK, scores:", scores, "idxs:", idxs, flush=True)

print(flush=True)
print("ALL STEPS PASSED -- no crash in this isolated script.", flush=True)
print("If app.py still segfaults but this script completes, the issue is", flush=True)
print("specific to how Streamlit loads/reruns things, not these libraries", flush=True)
print("in isolation -- a different, narrower investigation.", flush=True)