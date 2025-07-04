import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForCausalLM
from peft import PeftModel
import evaluate
import gc
from scipy import stats
import sys
import traceback


class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy types. """

    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# --- Configuration ---
BASE_MODEL_PATH = "base_model_path"
LORA_MODEL_PATH = "LoRA_model_path"
EVAL_DATA_PATH = "Eval_data_path"
OUTPUT_DIR = "analysis_results"

# --- Core analysis parameters ---
NUM_TOP_ACTIVATED_TO_SILENCE = 5
NUM_TOP_SUPPRESSED_TO_SILENCE = 1

# --- Script Start ---
torch._dynamo.config.disable = True
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(IMAGES_DIR, exist_ok=True)

try:
    bleu = evaluate.load("bleu")
except Exception as e:
    print(f"Could not load BLEU metric: {e}. Check internet connection.")
    sys.exit(1)


def cleanup_gpu():
    """Frees up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_eval_data():
    """Loads the evaluation dataset."""
    print(f"📄 Loading evaluation data from {EVAL_DATA_PATH}...")
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa_pairs = data.get('qa_pairs', [])
    if not qa_pairs or not all('answer' in qa for qa in qa_pairs):
        raise ValueError("Evaluation data is missing or malformed.")
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
    return qa_pairs


def get_last_mlp_layer(model):
    """Gets the final MLP layer from the model, handling PeftModel wrappers."""
    model_to_probe = model.base_model.model if isinstance(model, PeftModel) else model
    return model_to_probe.model.layers[-1].mlp


class SilenceHook:
    """A forward hook to zero out ('silence') specific neuron activations."""

    def __init__(self, neuron_indices):
        self.indices = [neuron_indices] if isinstance(neuron_indices, int) else neuron_indices
        print(f"🤫 SilenceHook initialized for neurons: {self.indices}")

    def __call__(self, module, p_input, p_output):
        output_tensor = p_output[0] if isinstance(p_output, tuple) else p_output
        if isinstance(output_tensor, torch.Tensor):
            if output_tensor.dim() == 3:
                output_tensor[:, :, self.indices] = 0.0
            elif output_tensor.dim() == 2:
                output_tensor[:, self.indices] = 0.0


def evaluate_performance(prediction, reference):
    """Calculates BLEU score for a generated answer."""
    length = len(prediction.split())
    try:
        bleu_score = bleu.compute(predictions=[prediction], references=[[reference]])['bleu']
    except Exception as e:
        print(f"⚠️ Warning: Could not compute BLEU score. {e}")
        bleu_score = 0.0

    return {'answer_length': length, 'bleu': bleu_score}


def analyze_model(model, tokenizer, qa_pairs, model_name):
    """Runs a model over the dataset, collecting performance and activations."""
    print(f"\n🔬 Analyzing model: {model_name}")
    model.eval()
    results, all_activations = [], []
    last_mlp_layer = get_last_mlp_layer(model)

    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc=f"Processing {model_name}", ncols=100):
            prompt = f"<bos><start_of_turn>user\n{qa['question']}<end_of_turn>\n<start_of_turn>model\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            captured_activations = []

            def hook_fn(module, p_input, p_output):
                act_tensor = p_input[0]
                if act_tensor.dim() == 3:
                    captured_activations.append(act_tensor[0, -1, :].detach().cpu().float())

            hook_handle = last_mlp_layer.register_forward_hook(hook_fn)
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            hook_handle.remove()

            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            results.append({
                'question': qa['question'],
                'reference_answer': qa['answer'],
                'generated_answer': generated_text,
                'performance': evaluate_performance(generated_text, qa['answer']),
            })

            if captured_activations:
                all_activations.append(torch.mean(torch.stack(captured_activations), dim=0).numpy())

            cleanup_gpu()

    print(f"✅ Analysis complete for {model_name}")
    return np.array(all_activations), results


def create_paper_visualizations(base_activations, lora_activations, top_indices):
    """Generates Figures 1 and 2 from the paper."""
    print("\n📊 Generating core paper visualizations (Figures 1, 2)...")
    activation_diff = np.mean(lora_activations, axis=0) - np.mean(base_activations, axis=0)
    top_changes = activation_diff[top_indices]

    # --- Figure 1: Bar chart of activation changes (Individual Figure) ---
    plt.figure(figsize=(12, 7))
    colors = ['red' if x > 0 else 'blue' for x in top_changes]
    plt.bar(range(len(top_indices)), top_changes, color=colors)
    plt.title(f'Figure 1: Top {len(top_indices)} Neuron Activation Changes', fontsize=16)
    plt.ylabel('Average Activation Change (LoRA - Base)')
    plt.xticks(range(len(top_indices)), [f'#{i}' for i in top_indices], rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_1_Activation_Change_Bars.png'), dpi=300)
    plt.close()
    print("  ✅ Figure 1 saved.")

    # --- Figure 2: Boxplot comparison (12 subplots in one Figure) ---
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), squeeze=False)
    fig.suptitle(f'Figure 2: Activation Distribution for Top {len(top_indices)} Neurons', fontsize=18)
    axes = axes.flatten()
    for i, neuron_idx in enumerate(top_indices):
        data = [base_activations[:, neuron_idx], lora_activations[:, neuron_idx]]
        bp = axes[i].boxplot(data, tick_labels=['Base', 'LoRA'], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[i].set_title(f'Neuron #{neuron_idx}\nΔ = {activation_diff[neuron_idx]:+.2f}')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots if top_indices has less than 12 items
    for i in range(len(top_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_2_Activation_Boxplots_Grid.png'), dpi=300)
    plt.close()
    print("  ✅ Figure 2 (subplot grid) saved.")


def create_silencing_visualizations(results_dict):
    """Generates Figure 3 from the paper (performance comparison) - now as individual figures."""
    print("📊 Generating silencing performance visualizations (Figure 3)...")
    models = list(results_dict.keys())
    metrics = {'bleu': 'BLEU Score', 'answer_length': 'Average Answer Length'}
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(models)))

    # Create individual figures for each metric
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(12, 8))
        avg_scores = [np.mean([r['performance'][metric_key] for r in results]) for model_name, results in
                      results_dict.items()]
        bars = plt.bar(models, avg_scores, color=colors)
        plt.title(f'Performance Comparison: {metric_name}', fontsize=16)
        plt.ylabel('Score' if 'Length' not in metric_name else 'Word Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.bar_label(bars, fmt='%.3f', fontsize=9, fontweight='bold')
        plt.tight_layout()

        # Save with appropriate filename
        filename = f'Figure_3_{metric_key.upper()}_Comparison.png'
        plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=300)
        plt.close()

    print("  ✅ Individual performance comparison figures saved.")


def create_statistical_analysis(results_dict):
    """Generates Figure 4 from the paper (statistical significance for BLEU)."""
    print("📊 Generating statistical significance visualization (Figure 4)...")
    metric = 'bleu'  # Only analyzing BLEU
    lora_scores = [r['performance'][metric] for r in results_dict.get('LoRA', [])]
    p_values_map = {}

    base_scores = [r['performance'][metric] for r in results_dict.get('Base', [])]
    if base_scores and lora_scores:
        p_values_map['Base vs LORA'] = stats.wilcoxon(lora_scores, base_scores, alternative='greater')[1]

    for name, results in results_dict.items():
        if 'Silenced' in name:
            label = f"LORA vs {name.replace('LoRA-', '')}"
            silenced_scores = [r['performance'][metric] for r in results]
            if silenced_scores and lora_scores:
                p_values_map[label] = stats.wilcoxon(lora_scores, silenced_scores, alternative='greater')[1]

    if not p_values_map: return

    plot_data = {k: -np.log10(max(v, 1e-50)) for k, v in p_values_map.items()}
    sorted_items = sorted(plot_data.items(), key=lambda item: item[1], reverse=True)
    labels, values = [item[0] for item in sorted_items], [item[1] for item in sorted_items]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(labels, values, color='royalblue', zorder=3)
    plt.gca().invert_yaxis()
    plt.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
    plt.title('Statistical Significance of Performance Differences (BLEU Score)', fontsize=16)
    plt.xlabel('-log10(p-value)')
    plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    plt.legend()
    plt.grid(axis='x', linestyle=':', alpha=0.7, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_4_Statistical_Significance.png'), dpi=300)
    plt.close()
    print("  ✅ Figure 4 saved.")


def main():
    print("=" * 60)
    print("🔬 Comprehensive Analysis Script for BWR-Specialized Model")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        qa_pairs = load_eval_data()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Initialization failed: {e}\nCheck model and data paths.");
        return

    try:
        # Step 1: Analyze Base and LoRA models
        print("\n--- Step 1: Analyzing Base and LoRA Models ---")
        base_model = Gemma3ForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
        base_activations, base_results = analyze_model(base_model, tokenizer, qa_pairs, "Base")
        del base_model;
        cleanup_gpu()

        lora_model_merged = PeftModel.from_pretrained(
            Gemma3ForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"),
            LORA_MODEL_PATH
        ).merge_and_unload()
        lora_activations, lora_results = analyze_model(lora_model_merged, tokenizer, qa_pairs, "LoRA")

        # Step 2: Identify key neurons for silencing
        print("\n--- Step 2: Identifying Key Neurons ---")
        activation_diff = np.mean(lora_activations, axis=0) - np.mean(base_activations, axis=0)
        sorted_indices = np.argsort(np.abs(activation_diff))[::-1]

        indices_for_viz = sorted_indices[:12]

        activated = [i for i in sorted_indices if activation_diff[i] > 0][:NUM_TOP_ACTIVATED_TO_SILENCE]
        suppressed = [i for i in sorted_indices if activation_diff[i] < 0][:NUM_TOP_SUPPRESSED_TO_SILENCE]
        indices_to_silence = activated + suppressed

        print(f"  - Top 12 neurons for visualization: {indices_for_viz.tolist()}")
        print(f"  - Neurons to be silenced ({len(indices_to_silence)}): {indices_to_silence}")

        # Step 3: Run silencing experiments
        print("\n--- Step 3: Running Neuron Silencing Experiments ---")
        results_dict = {"Base": base_results, "LoRA": lora_results}
        last_mlp_layer = get_last_mlp_layer(lora_model_merged)

        for neuron_idx in indices_to_silence:
            model_name = f"LoRA-Silenced-#{neuron_idx}"
            hook_handle = last_mlp_layer.register_forward_hook(SilenceHook(neuron_idx))
            _, silenced_res = analyze_model(lora_model_merged, tokenizer, qa_pairs, model_name)
            results_dict[model_name] = silenced_res
            hook_handle.remove()

        model_name = f"LoRA-Silenced-Key{len(indices_to_silence)}"
        hook_handle = last_mlp_layer.register_forward_hook(SilenceHook(indices_to_silence))
        _, silenced_res = analyze_model(lora_model_merged, tokenizer, qa_pairs, model_name)
        results_dict[model_name] = silenced_res
        hook_handle.remove()

        del lora_model_merged;
        cleanup_gpu()

        # Step 4: Generate all visualizations and save results
        print("\n--- Step 4: Generating Outputs ---")
        create_paper_visualizations(base_activations, lora_activations, indices_for_viz)
        create_silencing_visualizations(results_dict)
        create_statistical_analysis(results_dict)

        summary_df = pd.DataFrame({
            model: {k: np.mean([p['performance'][k] for p in res]) for k in res[0]['performance'].keys()}
            for model, res in results_dict.items()
        }).T
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_performance_metrics.csv'))
        print(f"\n💾 Summary metrics saved to {os.path.join(OUTPUT_DIR, 'summary_performance_metrics.csv')}")

        with open(os.path.join(OUTPUT_DIR, 'full_results_bleu_only.json'), 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, cls=NumpyEncoder, indent=2)
        print(f"💾 Full results saved to {os.path.join(OUTPUT_DIR, 'full_results_bleu_only.json')}")

        print("\n🎉 All analyses completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during the main execution: {e}")
        traceback.print_exc()
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()