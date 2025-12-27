import sys
import os
from pathlib import Path

# Add src to path if not installed
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
print(f"Added {src_path} to sys.path")

from spectral_trust import GSPDiagnosticsFramework, GSPConfig
import torch

def main():
    print("Checking CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "meta-llama/Llama-3.2-1B"
    print(f"Loading model: {model_name}")
    
    config = GSPConfig(
        model_name=model_name,
        device=device,
        save_plots=True,
        output_dir="./demo_results_script",
        model_kwargs={"attn_implementation": "eager"}
    )

    text = "The capital of France is Paris."
    print(f"Analyzing text: {text}")

    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            results = framework.analyze_text(text)
            
            print(f"Analyzed: {text}")
            print("\nFirst 5 layers energy:")
            for diag in results['layer_diagnostics'][:5]:
                print(f"Layer {diag.layer}: {diag.energy:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
