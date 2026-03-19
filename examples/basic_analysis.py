# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import sys
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Configuration
    model_name = "llama-3.2-1b" # Use a small model for the example
    
    print(f"Loading {model_name}...")
    config = GSPConfig(
        model_name=model_name,
        device="cuda", # Automatically falls back to CPU if no CUDA
        output_dir="./example_results"
    )
    
    # Initialize framework
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model("meta-llama/Llama-3.2-1B")
        
        # Text to analyze
        text = "The capital of France is Paris."
        print(f"Analyzing: '{text}'")
        
        # Run analysis
        results = framework.analyze_text(text)
        
        print("\nAnalysis Results:")
        print(f"{'Layer':>5} {'Energy':>10} {'HFER':>10}")
        print("-" * 30)
        
        for diag in results['layer_diagnostics']:
            print(f"{diag.layer:5d} {diag.energy:10.4f} {diag.hfer:10.4f}")
            
        print(f"\nFull results saved to {config.output_dir}")

if __name__ == "__main__":
    main()
