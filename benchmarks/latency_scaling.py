
import time
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

def run_latency_benchmark(model_name="gpt2", output_file="latency_scaling.png"):
    """
    Benchmark spectral analysis latency vs sequence length
    """
    print(f"Loading model {model_name} for benchmarking...")
    
    # Configure for speed
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=False,
        save_plots=False,  # Don't save per-run plots
        save_activations=False
    )
    
    framework = GSPDiagnosticsFramework(config)
    framework.instrumenter.load_model(model_name)
    
    # Sequence lengths to test
    # seq_lengths = [128, 256, 512, 1024]
    seq_lengths = [32, 64, 128, 256, 512] # Smaller range for quick CI, extend for prod
    
    framework.instrumenter.config.max_length = max(seq_lengths) + 50
    
    latencies = []
    
    print("Starting benchmark loop...")
    for seq_len in tqdm(seq_lengths):
        # Generate dummy text of approx length (repeat "the " N times)
        text = "word " * seq_len
        
        # We need to ensure token count is close to seq_len
        # So we clip/pad in the instrumenter or just rely on max_length
        # Ideally we want exact measurement.
        
        # Warmup
        if seq_len == seq_lengths[0]:
            try:
                framework.analyze_text(text[:100], save_results=False)
            except:
                pass
                
        start_time = time.perf_counter()
        
        # Run analysis (without saving results to disk to measure pure compute)
        try:
            # Force truncation in tokenizer call is handled by config.max_length usually,
            # but we want to vary input length.
            # We construct a text that is long enough.
            long_text = "test " * seq_len
            
            # Monkey-patch config momentarily if needed or just rely on input size
            res = framework.analyze_text(long_text, save_results=False)
            token_count = len(res['tokens'])
        except Exception as e:
            print(f"Error at len {seq_len}: {e}")
            token_count = seq_len
            
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        latencies.append(duration)
        print(f"Len: {token_count} tokens -> {duration:.4f}s")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, latencies, 'b-o', list(map(lambda x: x/60, seq_lengths)), 'r--') # Red dashed is linear ref? No, just plot points.
    plt.plot(seq_lengths, latencies, 'bo-', linewidth=2, label='Spectral Analysis Time')
    
    # Fit quadratic/cubic trend? Spectral analysis is O(N^3) for dense eig, O(N*k) for sparse
    # Let's just plot.
    
    plt.title(f"Spectral Analysis Latency vs Sequence Length\n({model_name})")
    plt.xlabel("Sequence Length (Tokens)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_file)
    print(f"Benchmark plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model to benchmark")
    parser.add_argument("--output", type=str, default="benchmarks/latency_scaling.png")
    args = parser.parse_args()
    
    run_latency_benchmark(args.model, args.output)
