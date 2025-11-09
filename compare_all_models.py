# import onnxruntime as ort
# import numpy as np
# import time
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# from transformers import AutoTokenizer
# import vllm  # Assuming vLLM python API


# def run_inference_onnx(session, input_feed, warmup=10, repetitions=50):
#     # Measure TTFT for ONNX
#     start_ttft = time.time()
#     session.run(None, input_feed)
#     end_ttft = time.time()
#     ttft = end_ttft - start_ttft

#     for _ in range(warmup): session.run(None, input_feed)
#     start = time.time()
#     for _ in range(repetitions):
#         session.run(None, input_feed)
#     end = time.time()
#     avg_latency = (end-start) / repetitions
#     throughput = list(input_feed.values())[0].shape[0] / avg_latency
#     return ttft, avg_latency, throughput

# def run_inference_tensorrt(engine_path, input_feed, batch_size=16, warmup=10, repetitions=50):
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())
#     context = engine.create_execution_context()

#     input_ids = input_feed['input_ids'].ravel().astype(np.int32)
#     attention_mask = input_feed['attention_mask'].ravel().astype(np.int32)

#     # Allocate device memory
#     d_input_ids = cuda.mem_alloc(input_ids.nbytes)
#     d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
#     output_size = 2048  # Adjust based on model output size
#     d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)

#     stream = cuda.Stream()

#     def infer():
#         cuda.memcpy_htod_async(d_input_ids, input_ids, stream)
#         cuda.memcpy_htod_async(d_attention_mask, attention_mask, stream)
#         context.execute_async_v2(bindings=[int(d_input_ids), int(d_attention_mask), int(d_output)], stream_handle=stream.handle)
#         stream.synchronize()

#     # TTFT
#     start_ttft = time.time()
#     infer()
#     end_ttft = time.time()

#     # Warmup
#     for _ in range(warmup):
#         infer()

#     # Timed runs
#     start = time.time()
#     for _ in range(repetitions):
#         infer()
#     end = time.time()

#     avg_latency = (end-start)/repetitions
#     throughput = batch_size / avg_latency
#     return start_ttft-end_ttft, avg_latency, throughput

# def run_inference_vllm(model_path, tokenizer, batch_size=16, sequence_length=256, warmup=5, repetitions=20):
#     from vllm import LLMEngine, SamplingParams
#     engine = LLMEngine(model=model_path)

#     inputs = ["The quick brown fox jumps over the lazy dog."] * batch_size

#     sampling_params = SamplingParams(max_tokens=sequence_length)

#     # Warming up
#     for _ in range(warmup):
#         list(engine.generate(inputs, sampling_params=sampling_params))

#     # TTFT timing (time for first token batch generation)
#     start_ttft = time.time()
#     list(engine.generate(inputs, sampling_params=sampling_params))
#     end_ttft = time.time()
#     ttft = end_ttft - start_ttft

#     # Timing repetitions
#     start = time.time()
#     for _ in range(repetitions):
#         list(engine.generate(inputs, sampling_params=sampling_params))
#     end = time.time()

#     avg_latency = (end - start) / repetitions
#     throughput = batch_size / avg_latency
#     return ttft, avg_latency, throughput


# def main():
#     batch_size = 16
#     sequence_length = 256
#     original_model_path = "/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512.onnx"
#     optimized_model_path = "//root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512_optimized.onnx"
#     tensorrt_engine_path = "/root/Shishir/model_bs16_seq512.trt"
#     vllm_model_path = "bert-large-uncased"  # or local vllm-supported model path

#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

#     # ONNX original
#     original_sess = ort.InferenceSession(original_model_path, sess_options, providers=providers)
#     input_feed = {
#         'input_ids': np.random.randint(0, 30522, size=(batch_size, sequence_length)).astype(np.int64),
#         'attention_mask': np.ones((batch_size, sequence_length)).astype(np.int64)
#     }
#     onnx_ttft, onnx_latency, onnx_throughput = run_inference_onnx(original_sess, input_feed)

#     # ONNX optimized
#     optimized_sess = ort.InferenceSession(optimized_model_path, sess_options, providers=providers)
#     optim_ttft, optim_latency, optim_throughput = run_inference_onnx(optimized_sess, input_feed)

#     # TensorRT inference
#     trt_ttft, trt_latency, trt_throughput = run_inference_tensorrt(tensorrt_engine_path, input_feed, batch_size)

#     # vLLM inference
#     tokenizer = AutoTokenizer.from_pretrained(vllm_model_path)
#     vllm_ttft, vllm_latency, vllm_throughput = run_inference_vllm(vllm_model_path, tokenizer, batch_size, sequence_length)

#     print("Latency Comparison:")
#     print(f"ONNX Original     TTFT: {onnx_ttft*1000:.2f} ms, Latency: {onnx_latency*1000:.2f} ms, Throughput: {onnx_throughput:.2f}")
#     print(f"ONNX Optimized    TTFT: {optim_ttft*1000:.2f} ms, Latency: {optim_latency*1000:.2f} ms, Throughput: {optim_throughput:.2f}")
#     print(f"TensorRT          TTFT: {trt_ttft*1000:.2f} ms, Latency: {trt_latency*1000:.2f} ms, Throughput: {trt_throughput:.2f}")
#     print(f"vLLM              TTFT: {vllm_ttft*1000:.2f} ms, Latency: {vllm_latency*1000:.2f} ms, Throughput: {vllm_throughput:.2f}")


# if __name__ == "__main__":
#     main()



# Create script with memory-efficient fixes
import onnxruntime as ort
import numpy as np
import time
import tensorrt as trt
import os
import gc
import torch

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def create_tensorrt_engine(onnx_path, engine_path, batch_size=16, seq_length=512, fp16_mode=True):
    """Convert ONNX to TensorRT engine with optimization profile"""
    print(f"\\nConverting {onnx_path} to TensorRT engine...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("✓ ONNX model parsed successfully")
    
    config = builder.create_builder_config()
    # REDUCED workspace size from 2GB to 512MB to avoid OOM
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)  # 512MB
    
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    else:
        print("✓ Using FP32 precision")
    
    # Add optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    min_shape = (1, seq_length)
    opt_shape = (batch_size, seq_length)
    max_shape = (batch_size * 2, seq_length)
    
    profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
    profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    print(f"✓ Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    print("Building TensorRT engine... This may take 2-5 minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved to {engine_path}")
    return True

def run_inference_onnx(session, input_feed, warmup=10, repetitions=50):
    """Run ONNX inference and measure metrics"""
    start_ttft = time.time()
    session.run(None, input_feed)
    ttft = time.time() - start_ttft

    for _ in range(warmup):
        session.run(None, input_feed)
    
    start = time.time()
    for _ in range(repetitions):
        session.run(None, input_feed)
    
    avg_latency = (time.time() - start) / repetitions
    batch_size = list(input_feed.values())[0].shape[0]
    throughput = batch_size / avg_latency
    
    return ttft, avg_latency, throughput

def run_inference_tensorrt(onnx_path, batch_size, seq_length, warmup=5, repetitions=20):
    """Run TensorRT inference with reduced iterations to save memory"""
    try:
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 536870912,  # 512MB instead of 2GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        print("  Creating ONNX Runtime session with TensorRT EP...")
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"  Active providers: {session.get_providers()}")
        
        input_feed = {
            'input_ids': np.random.randint(0, 30522, size=(batch_size, seq_length)).astype(np.int64),
            'attention_mask': np.ones((batch_size, seq_length)).astype(np.int64)
        }
        
        start_ttft = time.time()
        session.run(None, input_feed)
        ttft = time.time() - start_ttft
        
        # Reduced warmup iterations
        for _ in range(warmup):
            session.run(None, input_feed)
        
        start = time.time()
        for _ in range(repetitions):
            session.run(None, input_feed)
        
        avg_latency = (time.time() - start) / repetitions
        throughput = batch_size / avg_latency
        
        return ttft, avg_latency, throughput
        
    except Exception as e:
        print(f"  ERROR: TensorRT inference failed: {e}")
        return None, None, None

def main():
    # REDUCED batch size from 16 to 8 to fit in GPU memory
    batch_size = 8  # Changed from 16
    sequence_length = 512
    
    original_model_path = "/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512.onnx"
    optimized_model_path = "/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512_optimized.onnx"
    tensorrt_engine_path = "/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs8_seq512.trt"
    
    print("="*70)
    print(" BERT INFERENCE BENCHMARK: ONNX vs TensorRT")
    print("="*70)
    print(f"\\nConfiguration:")
    print(f"  Batch Size: {batch_size} (REDUCED for memory)")
    print(f"  Sequence Length: {sequence_length}")
    print(f"  Warmup Iterations: 10 (ONNX), 5 (TensorRT)")
    print(f"  Benchmark Iterations: 50 (ONNX), 20 (TensorRT)")
    
    print(f"\\nChecking model files...")
    if not os.path.exists(original_model_path):
        print(f"ERROR: Original model not found at {original_model_path}")
        return
    print(f"  ✓ Original ONNX model found")
    
    if not os.path.exists(optimized_model_path):
        print(f"ERROR: Optimized model not found at {optimized_model_path}")
        return
    print(f"  ✓ Optimized ONNX model found")
    
    # Clear GPU memory before starting
    print("\\nClearing GPU memory...")
    clear_gpu_memory()
    
    # Create TensorRT engine if missing
    if not os.path.exists(tensorrt_engine_path):
        print(f"\\n⚠ TensorRT engine not found. Creating it now...")
        success = create_tensorrt_engine(original_model_path, tensorrt_engine_path, 
                                        batch_size, sequence_length, fp16_mode=True)
        if not success:
            print("  WARNING: TensorRT engine creation failed. Skipping TensorRT benchmark.")
    else:
        print(f"  ✓ TensorRT engine already exists")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    input_feed = {
        'input_ids': np.random.randint(0, 30522, size=(batch_size, sequence_length)).astype(np.int64),
        'attention_mask': np.ones((batch_size, sequence_length)).astype(np.int64)
    }
    
    print("\\n" + "="*70)
    print(" RUNNING BENCHMARKS")
    print("="*70)
    
    results = {}
    
    # 1. ONNX Original
    print("\\n[1/3] Benchmarking ONNX Original...")
    try:
        clear_gpu_memory()
        original_sess = ort.InferenceSession(original_model_path, sess_options, providers=providers)
        print(f"  Active providers: {original_sess.get_providers()}")
        ttft, lat, tput = run_inference_onnx(original_sess, input_feed)
        results['ONNX Original'] = {'ttft': ttft, 'latency': lat, 'throughput': tput}
        print(f"  ✓ TTFT: {ttft*1000:.2f} ms | Latency: {lat*1000:.2f} ms | Throughput: {tput:.2f} samples/sec")
        del original_sess
        clear_gpu_memory()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 2. ONNX Optimized
    print("\\n[2/3] Benchmarking ONNX Optimized...")
    try:
        clear_gpu_memory()
        optimized_sess = ort.InferenceSession(optimized_model_path, sess_options, providers=providers)
        print(f"  Active providers: {optimized_sess.get_providers()}")
        ttft, lat, tput = run_inference_onnx(optimized_sess, input_feed)
        results['ONNX Optimized'] = {'ttft': ttft, 'latency': lat, 'throughput': tput}
        print(f"  ✓ TTFT: {ttft*1000:.2f} ms | Latency: {lat*1000:.2f} ms | Throughput: {tput:.2f} samples/sec")
        del optimized_sess
        clear_gpu_memory()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 3. TensorRT
    print("\\n[3/3] Benchmarking TensorRT...")
    clear_gpu_memory()
    ttft, lat, tput = run_inference_tensorrt(original_model_path, batch_size, sequence_length)
    if ttft:
        results['TensorRT'] = {'ttft': ttft, 'latency': lat, 'throughput': tput}
        print(f"  ✓ TTFT: {ttft*1000:.2f} ms | Latency: {lat*1000:.2f} ms | Throughput: {tput:.2f} samples/sec")
    
    # Results
    print("\\n" + "="*70)
    print(" BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"\\n{'Backend':<25} {'TTFT (ms)':<15} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-"*70)
    
    for backend, metrics in results.items():
        print(f"{backend:<25} {metrics['ttft']*1000:>14.2f} {metrics['latency']*1000:>14.2f} {metrics['throughput']:>14.2f}")
    
    # Speedup
    if 'ONNX Original' in results:
        baseline = results['ONNX Original']['latency']
        
        print("\\n" + "="*70)
        print(" SPEEDUP ANALYSIS")
        print("="*70)
        
        if 'ONNX Optimized' in results:
            speedup = baseline / results['ONNX Optimized']['latency']
            print(f"  ONNX Optimization: {speedup:.2f}x faster")
        
        if 'TensorRT' in results:
            speedup = baseline / results['TensorRT']['latency']
            print(f"  TensorRT:          {speedup:.2f}x faster")
    
    print("\\n" + "="*70)
    print(" BENCHMARK COMPLETE")
    print("="*70)
    print("\\nNote: Batch size reduced to 8 to fit in GPU memory")

if __name__ == "__main__":
    main()


# with open('/tmp/benchmark_memory_fixed.py', 'w') as f:
#     f.write(memory_fixed_script)

# print("✓ Memory-optimized script created!")
# print("\nKEY FIXES for OOM error:")
# print("1. Reduced batch size: 16 → 8")
# print("2. Reduced workspace: 2GB → 512MB")
# print("3. Added GPU memory clearing between benchmarks")
# print("4. Reduced TensorRT iterations: 50 → 20")
# print("5. Delete sessions after use")
# print("\nCopy and run:")
# print("cp /tmp/benchmark_memory_fixed.py /root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/benchmark.py")
# print("cd /root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/")
# print("python benchmark.py")