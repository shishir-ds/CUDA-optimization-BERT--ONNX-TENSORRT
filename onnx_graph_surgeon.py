import onnx_graphsurgeon as gs
import onnx
import glob

onnx_files = glob.glob("/root/Shishir/CUDA-optimization-BERT--ONNX-TENSORRT/model_bs16_seq512.onnx")
print("Found ONNX files:", onnx_files)

for onnx_path in onnx_files:
    print(f"\nProcessing: {onnx_path}")
    # Load original model
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    
    # Print summary of original model
    print(f"Original node count: {len(graph.nodes)}")
    orig_ops = set(node.op for node in graph.nodes)
    print(f"Original ops: {orig_ops}")

    # Save a readable summary (optional)
    with open(onnx_path.replace(".onnx", "_summary.txt"), "w") as f:
        for node in graph.nodes:
            f.write(f"{node.op} | {node.name} | inputs: {[i.name for i in node.inputs]}, outputs: {[o.name for o in node.outputs]}\n")
    
    # Remove all Dropout ops
    graph.nodes = [node for node in graph.nodes if node.op != "Dropout"]
    
    # Cleanup and topologically sort
    graph.cleanup().toposort()
    
    # Print summary of optimized model
    print(f"Optimized node count: {len(graph.nodes)}")
    opt_ops = set(node.op for node in graph.nodes)
    print(f"Optimized ops: {opt_ops}")

    # Export to ONNX and set IR version to 11 for compatibility
    optimized_model = gs.export_onnx(graph)
    optimized_model.ir_version = 11  # Set IR version here
    
    # Save optimized model
    opt_path = onnx_path.replace(".onnx", "_optimized.onnx")
    onnx.save(optimized_model, opt_path)

    # Save optimized model summary
    with open(opt_path.replace(".onnx", "_summary.txt"), "w") as f:
        for node in graph.nodes:
            f.write(f"{node.op} | {node.name} | inputs: {[i.name for i in node.inputs]}, outputs: {[o.name for o in node.outputs]}\n")
    
    print(f"Saved: {opt_path} and summaries for comparison.")
