
# import argparse
# import torch
# import onnxruntime as ort
# import numpy as np
# import time
# import os
# import threading
# from transformers import AutoTokenizer, AutoModel


# try:
#     import pynvml
#     pynvml.nvmlInit()
#     NVML_AVAILABLE = True
# except Exception:
#     NVML_AVAILABLE = False


# def get_gpu_stats(device_index=0):
#     if not NVML_AVAILABLE:
#         return None, None
#     try:
#         handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
#         util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#         mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
#         return util, mem_used
#     except Exception:
#         return None, None


# def stress_gpu(model, inputs, device, duration=10):
#     start_time = time.time()
#     utils, mems = [], []
#     torch.cuda.reset_peak_memory_stats()
#     model.eval()
#     with torch.no_grad():
#         while time.time() - start_time < duration:
#             try:
#                 _ = model(**inputs)
#             except Exception as e:
#                 print(f"Error during PyTorch inference: {e}")
#                 break
#             torch.cuda.synchronize()
#             util, mem = get_gpu_stats()
#             if util is not None:
#                 utils.append(util)
#             if mem is not None:
#                 mems.append(mem)
#     mean_util = np.mean(utils) if utils else None
#     peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
#     return mean_util, peak_mem


# def stress_gpu_onnx(session, inputs, duration=10):
#     start_time = time.time()
#     utils, mems = [], []
#     try:
#         torch.cuda.reset_peak_memory_stats()
#     except Exception:
#         pass
#     while time.time() - start_time < duration:
#         try:
#             session.run(None, inputs)
#         except Exception as e:
#             print(f"Error during ONNXRuntime inference: {e}")
#             break
#         util, mem = get_gpu_stats()
#         if util is not None:
#             utils.append(util)
#         if mem is not None:
#             mems.append(mem)
#     mean_util = np.mean(utils) if utils else None
#     peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
#     return mean_util, peak_mem


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default="bert-large-uncased")
#     parser.add_argument("--static-shape", action="store_true")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model = AutoModel.from_pretrained(args.model).to(device)

#     # Use longer input and large batch to ensure GPU workload
#     text = ["The quick brown fox jumps over the lazy dog."] * 256
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     print("\nRunning PyTorch GPU stress...")
#     mean_util, peak_mem = stress_gpu(model, inputs, device, duration=8)

#     util_str = f"{mean_util:.1f}%" if mean_util is not None else "N/A"
#     mem_str = f"{peak_mem:.1f} MB" if peak_mem is not None else "N/A"
#     print(f"PyTorch Mean GPU Utilization: {util_str} | Peak CUDA Memory: {mem_str}")

#     # Export ONNX with larger batch/static shape if desired
#     onnx_path = "bert_large_bs256.onnx"
#     sess_options = ort.SessionOptions()
#     sess_options.log_severity_level = 1  # reduce log noise
#     dynamic = not args.static_shape
#     torch.onnx.export(
#         model,
#         (inputs["input_ids"], inputs["attention_mask"]),
#         onnx_path,
#         input_names=["input_ids", "attention_mask"],
#         output_names=["output"],
#         dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}} if dynamic else None,
#         opset_version=17,
#     )

#     providers = [
#         ("ONNX-CUDA", "CUDAExecutionProvider"),
#         # ("ONNX-TensorRT", ("TensorrtExecutionProvider", {
#         #     "trt_engine_cache_enable": True,
#         #     "trt_engine_cache_path": "./trt_cache"
#         # })),
#     ]
#     onnx_inputs = {
#         "input_ids": inputs["input_ids"].cpu().numpy(),
#         "attention_mask": inputs["attention_mask"].cpu().numpy(),
#     }

#     for name, provider in providers:
#         print(f"\nRunning {name} GPU stress...")
#         try:
#             sess = ort.InferenceSession(
#                 onnx_path,
#                 sess_options=sess_options,
#                 providers=[provider] if isinstance(provider, str) else [provider]
#             )
#             print("Session Providers:", sess.get_providers())
#             mean_util, peak_mem = stress_gpu_onnx(sess, onnx_inputs, duration=8)
#             print(f"{name} Mean GPU Utilization: {mean_util:.1f}% | Peak CUDA Memory: {peak_mem:.1f} MB")
#         except Exception as e:
#             print(f"⚠️ Skipped {name} ({e})")
#             # Fallback if TensorRT fails
#             if name == "ONNX-TensorRT":
#                 print("Retrying ONNX without TensorRT...")
#                 try:
#                     sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
#                     mean_util, peak_mem = stress_gpu_onnx(sess, onnx_inputs, duration=8)
#                     print(f"ONNX-CUDA (fallback) Mean GPU Utilization: {mean_util:.1f}% | Peak CUDA Memory: {peak_mem:.1f} MB")
#                 except Exception as e2:
#                     print(f"⚠️ Fallback ONNX-CUDA also failed: {e2}")

#     # Summary printing and plotting left unchanged, add if needed

# if __name__ == "__main__":
#     main()


import argparse
import torch
import onnxruntime as ort
import numpy as np
import time
import matplotlib.pyplot as plt
import pynvml

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

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    text = ["The quick brown fox jumps over the lazy dog."] * 16
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    batch_sizes = [1, 8, 16]
    results = {k: [] for k in ["Backend", "Batch", "Latency (ms)", "Memory (MB)", "GPU Util (%)", "GPU Mem (MB)"]}

    for bs in batch_sizes:
        print(f"\n📦 Benchmarking batch size = {bs}")
        batch_inputs = {k: v[:bs] for k, v in inputs.items()}

        # PyTorch benchmark
        mean, std, mem, util, gmem = benchmark_pytorch(model, batch_inputs)
        print(f"🟢 PyTorch: {mean:.2f} ms ±{std:.2f}, {mem:.1f} MB, GPU {util or 0:.1f}% / {gmem or 0:.1f} MB")
        results["Backend"].append("PyTorch")
        results["Batch"].append(bs)
        results["Latency (ms)"].append(mean)
        results["Memory (MB)"].append(mem)
        results["GPU Util (%)"].append(util or 0)
        results["GPU Mem (MB)"].append(gmem or 0)

        # Export ONNX model for given batch size
        onnx_path = f"model_bs{bs}.onnx"
        dynamic = not args.static_shape
        torch.onnx.export(
            model,
            (batch_inputs["input_ids"], batch_inputs["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}} if dynamic else None,
            opset_version=17,
        )

        # ONNX providers to benchmark
        providers = [
            ("ONNX-CUDA", "CUDAExecutionProvider"),
            ("ONNX-TensorRT", ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache"
            })),
        ]

        onnx_inputs = {
            "input_ids": batch_inputs["input_ids"].cpu().numpy(),
            "attention_mask": batch_inputs["attention_mask"].cpu().numpy(),
        }

        for name, provider in providers:
            print(f"\nRunning {name} GPU stress...")
            try:
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 1
                sess = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[provider] if isinstance(provider, str) else [provider]
                )
                print("Session Providers:", sess.get_providers())
                mean, std, mem, util, gmem = benchmark_onnx(sess, onnx_inputs)
                print(f"{name}: {mean:.2f} ms ±{std:.2f}, {mem or 0:.1f} MB, GPU {util or 0:.1f}% / {gmem or 0:.1f} MB")
                results["Backend"].append(name)
                results["Batch"].append(bs)
                results["Latency (ms)"].append(mean)
                results["Memory (MB)"].append(mem or 0)
                results["GPU Util (%)"].append(util or 0)
                results["GPU Mem (MB)"].append(gmem or 0)
            except Exception as e:
                print(f"⚠️ Skipped {name} ({e})")
                if name == "ONNX-TensorRT":
                    print("Retrying ONNX without TensorRT...")
                    try:
                        sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CUDAExecutionProvider"])
                        mean, std, mem, util, gmem = benchmark_onnx(sess, onnx_inputs)
                        print(f"ONNX-CUDA (fallback) Mean GPU Utilization: {util:.1f}% | Peak CUDA Memory: {gmem:.1f} MB")
                        results["Backend"].append("ONNX-CUDA (fallback)")
                        results["Batch"].append(bs)
                        results["Latency (ms)"].append(mean)
                        results["Memory (MB)"].append(mem or 0)
                        results["GPU Util (%)"].append(util or 0)
                        results["GPU Mem (MB)"].append(gmem or 0)
                    except Exception as e2:
                        print(f"⚠️ Fallback ONNX-CUDA also failed: {e2}")

    # Summary print
    print("\n📊 Final Summary:")
    print(f"{'Backend':<20}{'Batch':<8}{'Latency (ms)':<15}{'Memory (MB)':<15}{'GPU Util(%)':<15}{'GPU Mem(MB)':<15}")
    for i in range(len(results["Backend"])):
        print(f"{results['Backend'][i]:<20}{results['Batch'][i]:<8}{results['Latency (ms)'][i]:<15.2f}"
              f"{results['Memory (MB)'][i]:<15.1f}{results['GPU Util (%)'][i]:<15.1f}{results['GPU Mem (MB)'][i]:<15.1f}")

    # Optional plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    backends = list(set(results["Backend"]))
    for backend in backends:
        xs = [results["Batch"][i] for i in range(len(results["Backend"])) if results["Backend"][i] == backend]
        ys = [results["Latency (ms)"][i] for i in range(len(results["Backend"])) if results["Backend"][i] == backend]
        plt.plot(xs, ys, marker="o", label=backend)
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("PyTorch vs ONNX vs TensorRT — Latency & GPU Utilization")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_gpu_results.png")
    print("\n📈 Saved plot as benchmark_gpu_results.png")


if __name__ == "__main__":
    main()

