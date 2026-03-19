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

import json
import os
import torch
import numpy as np
from pathlib import Path
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

class DiagnosisEngine:
    def __init__(self, model_name):
        self.model_name = model_name
        self.config = GSPConfig(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="float32",
            normalization="sym",
            save_activations=False
        )
        self.framework = GSPDiagnosticsFramework(self.config)
        self.results = {}
        
    def load_probes(self):
        # Locate probes.json relative to this file
        current_dir = Path(__file__).parent
        probe_path = current_dir / "data" / "probes.json"
        with open(probe_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def measure(self, text, tag):
        print(f"  Scanning: '{text[:40]}...'")
        res = self.framework.analyze_text(text, save_results=False)
        metrics = res['layer_diagnostics']
        
        # Store full traces
        trace = {
            "fiedler": [m.fiedler_value for m in metrics],
            "smoothness": [m.smoothness_index for m in metrics],
            "entropy": [m.spectral_entropy for m in metrics],
            "hfer": [m.hfer for m in metrics],
            "energy": [m.energy for m in metrics]
        }
        self.results[tag] = trace
        return trace

    def detect_anomalies(self, trace, metric_name="fiedler"):
        values = np.array(trace[metric_name])
        anomalies = []
        
        # 1. Instability (Huge Delta)
        deltas = np.diff(values)
        max_delta = np.max(np.abs(deltas)) if len(deltas) > 0 else 0
        if max_delta > 0.5:
             idx = np.argmax(np.abs(deltas))
             anomalies.append(f"INSTABILITY: Huge jump ({max_delta:.2f}) at Layer {idx}->{idx+1}")

        # 2. Lobotomy (Plateau)
        # Check for 5+ layers with variance < 0.001
        for i in range(len(values) - 5):
            window = values[i:i+5]
            if np.std(window) < 0.001:
                anomalies.append(f"LOBOTOMY: Plateau detected layers {i}-{i+5}")
                break # Report once per Metric
                
        # 3. Collapse (Near Zero)
        if np.min(values) < 0.05:
            anomalies.append(f"COLLAPSE: Values verify low (<0.05). Min: {np.min(values):.3f}")

        return anomalies

    def compare_traces(self, trace_a, trace_b):
        # Compute MSE between two traces (Understanding Gap)
        # We focus on Fiedler as primary structural metric
        a = np.array(trace_a['fiedler'])
        b = np.array(trace_b['fiedler'])
        
        # Ensure lengths match (truncate to shorter)
        n = min(len(a), len(b))
        mse = np.mean((a[:n] - b[:n])**2)
        return mse

    def check_heuristics(self):
        r = self.results
        report = []
        warnings = []
        
        # --- 1. Dynamic Signature Analysis (General Health) ---
        base_trace = r['standard_english']
        fiedler_anomalies = self.detect_anomalies(base_trace, "fiedler")
        
        if fiedler_anomalies:
            report.append(f"PATHOLOGY DETECTED (Baseline):")
            for a in fiedler_anomalies:
                report.append(f"  -> {a}")
                warnings.append(a.split(':')[0])

        # --- 2. Impact Test (Active vs Passive) ---
        # Look for scars (Pattern: Drop in early layers)
        active = np.array(r['active']['fiedler'])
        passive = np.array(r['passive']['fiedler'])
        diff = active - passive
        
        # Check layers 2-5 specifically (user hint)
        active_early = active[2:6]
        passive_early = passive[2:6]
        diff_early = active_early - passive_early
        early_diff = np.mean(diff_early) if len(diff) > 5 else 0
        
        print(f"    -> [Debug] Layer 2-5 Active: {np.round(active_early, 3)}")
        print(f"    -> [Debug] Layer 2-5 Passive: {np.round(passive_early, 3)}")
        print(f"    -> [Debug] Layer 2-5 Delta:   {np.round(diff_early, 3)} (Mean: {early_diff:.3f})")
        
        if early_diff > 0.05:
            msg = f"SIGNATURE: [Synthetic Scar] (Connectivity Drop)"
            desc = f"  -> Passive voice degrades structure in layers 2-5 (Mean Delta: {early_diff:.3f})."
            report.append(msg)
            report.append(desc)
            warnings.append("Synthetic Scar")

        # --- 3. Style Understanding Gap ---
        if 'natural' in r:
            gap = self.compare_traces(r['natural'], r['technical'])
            if gap > 0.05: # Threshold for MSE
                 msg = f"SIGNATURE: [Understanding Gap]"
                 desc = f"  -> Model treats Technical vs Natural language as disjoint concepts (MSE: {gap:.3f})"
                 report.append(msg)
                 report.append(desc)
                 warnings.append("Understanding Gap")

        # --- 4. Fragmented Baseline Check ---
        avg_base = np.mean(base_trace['fiedler'])
        if avg_base < 0.25:
             msg = "SIGNATURE: [Fragmented Baseline]"
             desc = f"  -> Global Fiedler is critically low ({avg_base:.2f}). Model may treat English as 'comment'."
             report.append(msg)
             report.append(desc)
             warnings.append("Fragmented Baseline")

        # --- 5. Multilingual Gearbox ---
        if 'ja_complex' in r:
            ja_entropy = np.mean(r['ja_complex']['entropy'])
            if ja_entropy < 0.2 and avg_base > 0.5:
                 msg = "CAPABILITY: [Adaptive Gearbox]"
                 report.append(msg)

        return report, warnings

    def run(self):
        print(f"Initializing Spectral Diagnosis for: {self.model_name}")
        self.framework.instrumenter.load_model(self.model_name)
        
        probes = self.load_probes()
        
        print("\nPhase 1: Impact Test (Active/Passive)")
        self.measure(probes['impact_test']['active'], 'active')
        self.measure(probes['impact_test']['passive'], 'passive')
        
        print("\nPhase 2: Baseline Test")
        self.measure(probes['baseline_test']['standard_english'], 'standard_english')
        
        print("\nPhase 3: Style Test (Natural/Technical)")
        if 'style_test' in probes:
            self.measure(probes['style_test']['natural'], 'natural')
            self.measure(probes['style_test']['technical'], 'technical')

        print("\nPhase 4: Multilingual Test")
        self.measure(probes['multilingual_test']['ja_complex'], 'ja_complex')
        
        print("\n" + "="*50)
        print(f"MEDICAL REPORT: {self.model_name}")
        print("="*50)
        
        report, warnings = self.check_heuristics()
        
        if not report:
            print("No specific pathology detected (Nominal).")
        else:
            for line in report:
                print(line)
        
        # Print Vital Signs (Min/Max)
        base = self.results['standard_english']['fiedler']
        print("-" * 50)
        print(f"VITAL SIGNS (Baseline Fiedler):")
        print(f"  Range: [{np.min(base):.3f}, {np.max(base):.3f}]")
        print(f"  Mean:  {np.mean(base):.3f}")
        print(f"  Vol (Std): {np.std(base):.3f}")

        print("-" * 50)
        if warnings:
            print(f"WARNINGS: {', '.join(warnings)}")
        else:
            print("Status: Healthy")
            
def run_diagnosis(model_name):
    engine = DiagnosisEngine(model_name)
    engine.run()

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "microsoft/Phi-3-mini-4k-instruct"
    run_diagnosis(model)
