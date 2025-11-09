import onnxruntime as ort
import numpy as np
import time

def run_inference_with_ttft(session, input_feed, warmup=10, repetitions=50):
    # Measure TTFT: time for the very first run (cold start)
    start_ttft = time.time()
    session.run(None, input_feed)
    end_ttft = time.time()
    ttft = end_ttft - start_ttft
    print("TTFT measured")

    # Warmup runs
    for _ in range(warmup):
        session.run(None, input_feed)
    print("warmup done")

    # Timing actual runs for average latency and throughput
    start = time.time()
    for _ in range(repetitions):
        session.run(None, input_feed)
    end = time.time()
    print("latency calculation done")

    avg_latency = (end - start) / repetitions
    throughput = list(input_feed.values())[0].shape[0] / avg_latency
    return ttft, avg_latency, throughput

# File paths to original and optimized ONNX models
original_model_path = "/root/Shishir/model_bs16_seq512.onnx"
optimized_model_path = "/root/Shishir/model_bs16_seq512_optimized.onnx"

# Create inference session options with all graph optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Use GPU if available, otherwise defaults to CPU execution provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Load the original and optimized ONNX models with specified providers
original_sess = ort.InferenceSession(original_model_path, sess_options, providers=providers)
optimized_sess = ort.InferenceSession(optimized_model_path, sess_options, providers=providers)

# Print input information for debugging
print("Original model inputs:")
for inp in original_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}, {inp.type}")

print("Optimized model inputs:")
for inp in optimized_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}, {inp.type}")

# Prepare dummy inputs matching model requirements
batch_size = 16
sequence_length = 256  # You can vary this up to 512 if your model supports dynamic axes

input_feed = {
    'input_ids': np.random.randint(0, 30522, size=(batch_size, sequence_length)).astype(np.int64),
    'attention_mask': np.ones((batch_size, sequence_length)).astype(np.int64)
}

print("Inference started for original model")
ttft_orig, latency_orig, throughput_orig = run_inference_with_ttft(original_sess, input_feed)

print("Inference started for optimized model")
ttft_opt, latency_opt, throughput_opt = run_inference_with_ttft(optimized_sess, input_feed)

print(f"Original ONNX Model - TTFT: {ttft_orig*1000:.2f} ms, Avg. Latency: {latency_orig*1000:.2f} ms, Throughput: {throughput_orig:.2f} samples/sec")
print(f"Optimized ONNX Model - TTFT: {ttft_opt*1000:.2f} ms, Avg. Latency: {latency_opt*1000:.2f} ms, Throughput: {throughput_opt:.2f} samples/sec")
