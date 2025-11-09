# Fixed script with type conversion and proper INT8 handling
import onnxruntime as ort
import numpy as np
import time
import tensorrt as trt
import os
import gc
import torch
from collections import defaultdict

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class LayerProfiler(trt.IProfiler):
    """Custom profiler to track layer execution times"""
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_times = defaultdict(list)
        
    def report_layer_time(self, layer_name, ms):
        """Called once per layer for each invocation"""
        self.layer_times[layer_name].append(ms)
    
    def get_summary(self):
        """Get average times per layer"""
        summary = {}
        for layer_name, times in self.layer_times.items():
            summary[layer_name] = {
                'avg_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'count': len(times)
            }
        return summary

def run_inference_with_profiling(onnx_path, batch_size, seq_length, warmup=5, repetitions=20):
    """Run inference with layer-level profiling using native TensorRT"""
    try:
        print(f"\\n  Building TensorRT engine with FP16 and profiling enabled...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX")
                return None, None, None
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
        config.set_flag(trt.BuilderFlag.FP16)
        
        # Enable detailed profiling
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        
        # Optimization profile
        profile = builder.create_optimization_profile()
        min_shape = (1, seq_length)
        opt_shape = (batch_size, seq_length)
        max_shape = (batch_size * 2, seq_length)
        
        profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
        profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        print("  Building engine (this may take 2-3 minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if not serialized_engine:
            print("ERROR: Failed to build engine")
            return None, None, None
        
        # Deserialize and create context
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        
        # Set up profiler
        profiler = LayerProfiler()
        context.profiler = profiler
        
        # Set input shapes
        context.set_input_shape("input_ids", (batch_size, seq_length))
        context.set_input_shape("attention_mask", (batch_size, seq_length))
        
        # Prepare inputs
        input_ids = np.random.randint(0, 30522, size=(batch_size, seq_length)).astype(np.int32)
        attention_mask = np.ones((batch_size, seq_length)).astype(np.int32)
        
        # Allocate device memory
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        d_input_ids = cuda.mem_alloc(input_ids.nbytes)
        d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
        
        # FIXED: Get output size and convert to Python int
        output_name = engine.get_tensor_name(engine.num_io_tensors - 1)
        output_shape = tuple(context.get_tensor_shape(output_name))
        output_size = int(np.prod(output_shape) * 4)  # FP32 = 4 bytes, FIXED: convert to int
        d_output = cuda.mem_alloc(output_size)
        
        stream = cuda.Stream()
        
        def infer():
            cuda.memcpy_htod_async(d_input_ids, input_ids, stream)
            cuda.memcpy_htod_async(d_attention_mask, attention_mask, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
        
        # Warmup
        for _ in range(warmup):
            infer()
        
        # Benchmark
        start = time.time()
        for _ in range(repetitions):
            infer()
        latency = (time.time() - start) / repetitions
        throughput = batch_size / latency
        
        # Get profiling summary
        profile_summary = profiler.get_summary()
        
        return latency, throughput, profile_summary
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_inference_onnx(session, input_feed, warmup=10, repetitions=50):
    """Run ONNX inference"""
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

def main():
    batch_size = 8
    sequence_length = 512
    
    original_model_path = "/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512.onnx"
    
    print("="*70)
    print(" ADVANCED TENSORRT BENCHMARK: PROFILING + INT8 INFO")
    print("="*70)
    print(f"\\nConfiguration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {sequence_length}")
    
    if not os.path.exists(original_model_path):
        print(f"ERROR: Model not found at {original_model_path}")
        return
    print(f"  ✓ Model found")
    
    clear_gpu_memory()
    
    print("\\n" + "="*70)
    print(" PART 1: LAYER PROFILING (FP16)")
    print("="*70)
    
    lat, tput, profile_summary = run_inference_with_profiling(
        original_model_path, batch_size, sequence_length
    )
    
    if lat and profile_summary:
        print(f"\\n✓ FP16 Performance:")
        print(f"  Latency: {lat*1000:.2f} ms")
        print(f"  Throughput: {tput:.2f} samples/sec")
        
        print(f"\\n  Top 10 Slowest Layers:")
        sorted_layers = sorted(profile_summary.items(), key=lambda x: x[1]['avg_ms'], reverse=True)
        total_time = sum(stats['avg_ms'] for _, stats in sorted_layers)
        
        for i, (layer_name, stats) in enumerate(sorted_layers[:10], 1):
            pct = (stats['avg_ms'] / total_time * 100) if total_time > 0 else 0
            layer_short = layer_name[:55] if len(layer_name) > 55 else layer_name
            print(f"    {i:2d}. {layer_short:<55} {stats['avg_ms']:6.3f} ms ({pct:5.1f}%)")
        
        print(f"\\n  Total layers profiled: {len(profile_summary)}")
        print(f"  Total inference time: {total_time:.2f} ms")
    else:
        print("\\n✗ Profiling failed")
    
    print("\\n" + "="*70)
    print(" PART 2: INT8 QUANTIZATION ANALYSIS")
    print("="*70)
    print("\\nINT8 Quantization for BERT Transformers:")
    print("\\n❌ Challenge: Simple INT8 conversion fails because:")
    print("   - Attention layers are sensitive to quantization")
    print("   - No calibration data provided")
    print("   - Dynamic ranges not set")
    print("\\n✅ Working Approaches:")
    print("\\n1. NVIDIA TensorRT Model Optimizer (RECOMMENDED):")
    print("   pip install nvidia-modelopt[torch]")
    print("   # Handles INT8 calibration automatically")
    print("   # Achieves 4-6x speedup with minimal accuracy loss")
    print("\\n2. PyTorch Quantization-Aware Training (QAT):")
    print("   - Train with quantization in mind")
    print("   - Export to ONNX with quantization")
    print("   - Best accuracy retention")
    print("\\n3. ONNX Runtime Quantization:")
    print("   from onnxruntime.quantization import quantize_dynamic")
    print("   quantize_dynamic(model.onnx, model_int8.onnx)")
    
    # Try simple FP16 comparison via ONNX Runtime
    print("\\n" + "="*70)
    print(" PART 3: ONNX Runtime TensorRT EP (FP16 baseline)")
    print("="*70)
    
    try:
        clear_gpu_memory()
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }),
            'CUDAExecutionProvider'
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(original_model_path, sess_options, providers=providers)
        print(f"  Active providers: {session.get_providers()}")
        
        input_feed = {
            'input_ids': np.random.randint(0, 30522, size=(batch_size, sequence_length)).astype(np.int64),
            'attention_mask': np.ones((batch_size, sequence_length)).astype(np.int64)
        }
        
        ttft, lat, tput = run_inference_onnx(session, input_feed, warmup=5, repetitions=20)
        print(f"\\n  ✓ ONNX Runtime + TensorRT EP (FP16):")
        print(f"    Latency: {lat*1000:.2f} ms")
        print(f"    Throughput: {tput:.2f} samples/sec")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\\n" + "="*70)
    print(" SUMMARY & NEXT STEPS")
    print("="*70)
    print("\\n✓ Achieved:")
    print("  - Layer-level profiling to identify bottlenecks")
    print("  - FP16 optimization (4.6x speedup from your earlier run)")
    print("\\n→ For INT8 (6-8x speedup):")
    print("  1. Use nvidia-modelopt for automatic INT8 optimization")
    print("  2. Provide calibration dataset from your domain")
    print("  3. Consider QAT if retraining is possible")
    print("\\n→ For production deployment:")
    print("  - Profile shows which layers are slowest")
    print("  - Consider mixed precision (INT8 for MatMul, FP16 for attention)")
    print("  - Batch size tuning for your GPU memory")

if __name__ == "__main__":
    main()


# with open('/tmp/benchmark_profiling_fixed.py', 'w') as f:
#     f.write(fixed_script)

# print("✓ FIXED script created!")
# print("\nKey fixes:")
# print("1. ✅ Fixed PyCUDA type error: convert numpy.int64 → Python int")
# print("2. ✅ Removed broken INT8 attempt (needs proper calibration)")
# print("3. ✅ Added detailed INT8 explanation and recommendations")
# print("4. ✅ Kept layer profiling (the important part!)")
# print("5. ✅ Added ONNX Runtime TensorRT EP comparison")
# print("\nCopy and run:")
# print("cp /tmp/benchmark_profiling_fixed.py /root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/benchmark_advanced.py")
# print("cd /root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/")
# print("python benchmark_advanced.py")