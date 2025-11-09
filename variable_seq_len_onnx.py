import argparse
import torch
import onnxruntime as ort
import numpy as np
import time
import pynvml
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

pynvml.nvmlInit()

def get_gpu_stats(device_index=0):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        return util, mem_used
    except Exception:
        return None, None

def benchmark_pytorch(model, inputs, runs=50):
    latencies = []
    utils, mems = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # Warmup
            _ = model(**inputs)
        torch.cuda.synchronize()
        for _ in range(runs):
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            end = time.time()
            util, mem = get_gpu_stats()
            latencies.append((end - start)*1000)
            if util is not None:
                utils.append(util)
            if mem is not None:
                mems.append(mem)
    return np.mean(latencies), np.std(latencies), torch.cuda.max_memory_allocated() / (1024**2), np.mean(utils), np.mean(mems)

def benchmark_onnx(session, inputs, runs=50):
    latencies = []
    utils, mems = [], []
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    for _ in range(10):  # Warmup
        session.run(None, inputs)
    for _ in range(runs):
        start = time.time()
        session.run(None, inputs)
        end = time.time()
        util, mem = get_gpu_stats()
        latencies.append((end - start)*1000)
        if util is not None:
            utils.append(util)
        if mem is not None:
            mems.append(mem)
    return np.mean(latencies), np.std(latencies), torch.cuda.max_memory_allocated() / (1024**2), np.mean(utils), np.mean(mems)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-large-uncased")
    parser.add_argument("--static-shape", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    # Use batch size 16 and max sequence length 512 (variable if static-shape not specified)
    batch_size = 16
    max_seq_len = 512

    # Example minimal input for export, batching full max len to allow dynamic length axis
    text = ["The quick brown fox jumps over the lazy dog."] * batch_size
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    onnx_path = f"model_bs{batch_size}_seq{max_seq_len}.onnx"

    dynamic = not args.static_shape

    # Export ONNX with dynamic sequence length axis (axis 1)
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "attention_mask": {0: "batch", 1: "seq_len"}} if dynamic else None,
        opset_version=17,
    )

    print(f"Exported ONNX model to {onnx_path} with dynamic sequence length: {dynamic}")

if __name__ == "__main__":
    main()
