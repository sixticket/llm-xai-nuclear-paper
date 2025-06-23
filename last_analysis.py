import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForCausalLM
from peft import PeftModel
import evaluate
import gc
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    """ NumPy 타입을 JSON으로 변환하기 위한 Custom Encoder """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 설정 (사용자 환경에 맞게 수정) ---
torch._dynamo.config.disable = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL_PATH = r"path/to/gemma-3-1b-it"
LORA_MODEL_PATH = r"path/to/output/lora/model"
EVAL_DATA_PATH = r"path/to/qa/eval_dataset"
OUTPUT_DIR = r"path/to/analysis/results"

# --- [수정] 분석의 핵심 파라미터 ---
NUM_TOP_ACTIVATED_TO_SILENCE = 5    # 집중 분석할 '증폭' 뉴런 개수
NUM_TOP_SUPPRESSED_TO_SILENCE = 1   # 집중 분석할 '억제' 뉴런 개수

# --- 코드 시작 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(IMAGES_DIR, exist_ok=True)

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()


def load_eval_data():
    print("📄 평가 데이터 로드 중...")
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa_pairs = data['qa_pairs']
    if not all('answer' in qa for qa in qa_pairs):
        raise ValueError("평가 데이터에 BLEU/ROUGE 계산을 위한 'answer' 키가 없습니다.")
    print(f"✅ {len(qa_pairs)}개 QA 쌍 로드 완료 (정답 포함)")
    return qa_pairs


def get_last_mlp_layer(model):
    if isinstance(model, PeftModel):
        return model.base_model.model.model.layers[-1].mlp
    else:
        return model.model.layers[-1].mlp


class SilenceHook:
    def __init__(self, neuron_indices_to_silence):
        if isinstance(neuron_indices_to_silence, int):
            self.indices = [neuron_indices_to_silence]
        else:
            self.indices = neuron_indices_to_silence
        print(f"🤫 SilenceHook: 뉴런 #{self.indices}의 활성화를 0으로 고정합니다.")

    def __call__(self, module, p_input, p_output):
        if isinstance(p_output, tuple):
            output_tensor = p_output[0]
        else:
            output_tensor = p_output
        if not isinstance(output_tensor, torch.Tensor):
            return None
        try:
            max_idx = output_tensor.shape[-1]
            valid_indices = [idx for idx in self.indices if idx < max_idx]
            if not valid_indices:
                return None
            if output_tensor.dim() == 3:
                output_tensor[:, :, valid_indices] = 0.0
            elif output_tensor.dim() == 2:
                output_tensor[:, valid_indices] = 0.0
        except Exception as e:
            print(f"⚠️ SilenceHook 오류: {e}")
        return None


def evaluate_performance(generated_answer, reference_answer):
    length = len(generated_answer.split())
    try:
        bleu_score = bleu.compute(predictions=[generated_answer], references=[[reference_answer]])['bleu']
        rouge_scores = rouge.compute(predictions=[generated_answer], references=[reference_answer])
    except Exception as e:
        print(f"⚠️ BLEU/ROUGE 계산 오류: {e}")
        bleu_score = 0.0
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    return {
        'answer_length': length,
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
    }


def analyze_model(model, tokenizer, qa_pairs, model_name):
    print(f"\n{'=' * 20}\n🔬 {model_name} 분석 시작 (평균 활성화 방식)...\n{'=' * 20}")
    model.eval()
    results = []
    all_activations = []
    last_mlp_layer = get_last_mlp_layer(model)

    with torch.no_grad():
        for i, qa in enumerate(tqdm(qa_pairs, desc=f"{model_name} 처리중", ncols=100)):
            question = qa['question']
            reference_answer = qa['answer']
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                captured_activations_list = []

                def hook_fn(module, p_input, p_output):
                    nonlocal captured_activations_list
                    try:
                        if isinstance(p_input, tuple) and len(p_input) > 0:
                            input_tensor = p_input[0]
                            if isinstance(input_tensor, torch.Tensor) and input_tensor.numel() > 0:
                                if input_tensor.device.type == 'cuda':
                                    torch.cuda.synchronize()
                                if input_tensor.dim() == 3:
                                    captured_activations_list.append(
                                        input_tensor[0, -1, :].detach().float().cpu().numpy())
                                elif input_tensor.dim() == 2:
                                    captured_activations_list.append(input_tensor[-1, :].detach().float().cpu().numpy())
                    except Exception as e:
                        print(f"⚠️ 활성화 추출 오류: {e}")

                hook_handle = last_mlp_layer.register_forward_hook(hook_fn)

                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
                )
                hook_handle.remove()

                new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                if '<end_of_turn>' in generated_answer:
                    generated_answer = generated_answer.split('<end_of_turn>')[0].strip()

                performance_metrics = evaluate_performance(generated_answer, reference_answer)

                mean_activations = None
                if captured_activations_list:
                    mean_activations = np.mean(captured_activations_list, axis=0)
                    all_activations.append(mean_activations)
                    top5_indices = np.argsort(mean_activations)[-5:][::-1]
                    top5_neurons = [{'neuron_idx': int(idx), 'activation': float(mean_activations[idx])} for idx in
                                    top5_indices]
                else:
                    try:
                        hidden_size = model.config.hidden_size
                    except AttributeError:
                        hidden_size = 2048
                    default_activations = np.zeros(hidden_size)
                    all_activations.append(default_activations)
                    top5_neurons = []

                results.append({
                    'question_idx': i,
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'performance': performance_metrics,
                    'top5_activated_neurons': top5_neurons
                })

            except Exception as e:
                print(f"⚠️ 질문 {i} 처리 중 오류: {e}")
                try:
                    hidden_size = model.config.hidden_size
                except AttributeError:
                    hidden_size = 2048
                all_activations.append(np.zeros(hidden_size))
                results.append({
                    'question_idx': i, 'question': question, 'reference_answer': reference_answer,
                    'generated_answer': "Error",
                    'performance': {k: 0.0 for k in ['answer_length', 'bleu', 'rouge1', 'rouge2', 'rougeL']},
                    'top5_activated_neurons': []
                })
            finally:
                if 'inputs' in locals():
                    del inputs
                cleanup_gpu()

    print(f"✅ {model_name} 분석 완료!")
    return np.array(all_activations), results


def create_paper_visualizations(base_activations, lora_activations, top_changed_indices):
    print("\n📊 논문 핵심 그림 재현 시작...")

    activation_diff = np.mean(lora_activations, axis=0) - np.mean(base_activations, axis=0)

    # --- Figure 1: Top 12 Neuron Changes ---
    num_to_plot_bar = len(top_changed_indices)
    plt.figure(figsize=(12, 7))
    top_changes = activation_diff[top_changed_indices]
    colors = ['red' if x > 0 else 'blue' for x in top_changes]
    plt.bar(range(num_to_plot_bar), top_changes, color=colors)
    plt.title(f'Figure 1: Top {num_to_plot_bar} Neuron Activation Changes', fontsize=16)
    plt.ylabel('Average Activation Change', fontsize=12)
    plt.xticks(range(num_to_plot_bar), [f'#{i}' for i in top_changed_indices], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_1_Top_Neuron_Changes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Figure 1: Top {num_to_plot_bar} Neuron Changes 저장 완료")

    # --- Figure 2: Detailed activation distributions (Top 12) ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Figure 2: Detailed Activation Distributions for Top 12 Neurons', fontsize=20)
    for i, neuron_idx in enumerate(top_changed_indices):
        ax = axes[i]
        base_n = base_activations[:, neuron_idx]
        lora_n = lora_activations[:, neuron_idx]
        ax.plot(base_n, color='blue', alpha=0.6, label='Base')
        ax.plot(lora_n, color='red', alpha=0.6, label='LoRA')
        ax.set_title(f'Neuron #{neuron_idx}\nΔ={activation_diff[neuron_idx]:+.2f}', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_2_Activation_Distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Figure 2: Top 12 Activation Distributions 저장 완료")

    # --- [수정] Figure 3: Boxplot comparison (Top 12로 통일) ---
    num_to_plot_box = len(top_changed_indices)  # 12개 모두 사용

    # 4x3 배열로 Figure 2와 통일
    nrows = 4
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 15), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f'Figure 3: Boxplot Comparison for Top {num_to_plot_box} Neurons', fontsize=20)

    for i, neuron_idx in enumerate(top_changed_indices):
        ax = axes[i]
        data_to_plot = [base_activations[:, neuron_idx], lora_activations[:, neuron_idx]]
        bp = ax.boxplot(data_to_plot, tick_labels=['Base', 'LoRA'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title(f'Neuron #{neuron_idx}\nΔ = {activation_diff[neuron_idx]:+.2f}', fontsize=12)
        ax.set_ylabel('Activation Value')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 만약의 경우를 대비해 남는 subplot 숨기기 (12개일 경우 필요 없음)
    for i in range(num_to_plot_box, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_3_Boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Figure 3: Top {num_to_plot_box} Boxplots 저장 완료")


def create_silencing_visualizations(results_dict):
    print("\n📊 뉴런 사일런싱 효과 분석 그림 생성 (개별 파일)...")
    metrics = ['bleu', 'rougeL', 'answer_length']
    metric_names = ['BLEU Score', 'Rouge-L Score', 'Answer Length']
    file_tags = ['BLEU', 'RougeL', 'AnswerLength']
    models = list(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for i, (metric, name, tag) in enumerate(zip(metrics, metric_names, file_tags)):
        plt.figure(figsize=(12, 7))
        avg_scores = [np.mean([r['performance'][metric] for r in results_dict[model]]) for model in models]
        bars = plt.bar(models, avg_scores, color=colors, alpha=0.8)
        plt.title(f'Comparison of Average {name}', fontsize=16)
        plt.ylabel('Score' if 'length' not in name.lower() else 'Word Count', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', va='bottom', ha='center',
                     fontweight='bold')
        plt.tight_layout()
        filename = f"Figure_4_{i + 1}_{tag}_Comparison.png"
        plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename} 저장 완료")


def create_statistical_analysis(results_dict):
    print("\n📊 통계적 유의성 검정 시작...")
    base_scores = {
        'bleu': [r['performance']['bleu'] for r in results_dict.get('Base', [])],
        'rougeL': [r['performance']['rougeL'] for r in results_dict.get('Base', [])]
    }
    lora_scores = {
        'bleu': [r['performance']['bleu'] for r in results_dict.get('LoRA', [])],
        'rougeL': [r['performance']['rougeL'] for r in results_dict.get('LoRA', [])]
    }

    if not base_scores['bleu'] or not lora_scores['bleu']:
        print("⚠️ Base 또는 LoRA 모델의 결과가 없어 통계 분석을 건너뜁니다.")
        return {}

    results = {}
    for metric in ['bleu', 'rougeL']:
        stat, p_val = stats.wilcoxon(lora_scores[metric], base_scores[metric], alternative='greater')
        results[f'Base_vs_LoRA_{metric}'] = {'p_value': p_val, 'significant': p_val < 0.05}

    for model_name in results_dict.keys():
        if 'Silenced' in model_name:
            for metric in ['bleu', 'rougeL']:
                silenced_scores = [r['performance'][metric] for r in results_dict[model_name]]
                if not silenced_scores: continue
                stat, p_val = stats.wilcoxon(lora_scores[metric], silenced_scores)
                results[f'LoRA_vs_{model_name}_{metric}'] = {'p_value': p_val, 'significant': p_val < 0.05}

    significant_tests = {k: v['p_value'] for k, v in results.items() if v['significant']}
    if not significant_tests:
        print("  - 통계적으로 유의미한 차이를 보인 테스트가 없습니다.")
        return results

    plt.figure(figsize=(12, max(6, len(significant_tests) * 0.5)))
    sorted_tests = sorted(significant_tests.items(), key=lambda item: item[1])
    test_names = [item[0] for item in sorted_tests]
    p_values = [item[1] for item in sorted_tests]

    bars = plt.barh(range(len(test_names)), [-np.log10(p) if p > 0 else 50 for p in p_values])
    plt.yticks(range(len(test_names)), test_names, rotation=0)
    plt.xlabel('-log10(p-value)')
    plt.title('Figure 5: Statistical Significance Test Results (p < 0.05)')
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'Figure_5_Statistical_Tests.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Figure 5 저장 완료")
    return results


def create_question_type_analysis(results_dict):
    # [수정] Figure 6을 각 유형별 개별 파일로 저장
    print("\n📊 질문 유형별 성능 분석 시작 (개별 파일 저장)...")

    def classify_question(question):
        question = question.lower()
        if any(word in question for word in ['what is', 'what are', 'who', 'where', 'when', 'define', 'list']):
            return 'Factual'
        elif any(word in question for word in ['how', 'why', 'explain']):
            return 'Explanatory'
        elif any(word in question for word in ['compare', 'difference', 'versus']):
            return 'Comparative'
        else:
            return 'Other'

    model_names = list(results_dict.keys())
    question_types = {}
    for i, result in enumerate(results_dict[model_names[0]]):
        q_type = classify_question(result['question'])
        if q_type not in question_types:
            question_types[q_type] = {model: [] for model in model_names}
        for model in model_names:
            question_types[q_type][model].append(results_dict[model][i]['performance']['bleu'])

    if not question_types:
        print("  - 분석할 질문 유형이 없습니다.")
        return {}

    for q_type, data in question_types.items():
        plt.figure(figsize=(10, 7))
        models = list(data.keys())
        avg_scores = [np.mean(data[model]) for model in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

        bars = plt.bar(models, avg_scores, alpha=0.7, color=colors)

        plt.title(f'Performance on {q_type} Questions (n={len(data[models[0]])})', fontsize=16)
        plt.ylabel('Average BLEU Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom',
                     fontweight='bold')

        plt.tight_layout()
        filename = f"Figure_6_{q_type}_Analysis.png"
        plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename} 저장 완료")

    return question_types


def create_answer_quality_examples(results_dict):
    print("\n📊 답변 품질 변화 예시 생성...")
    if 'LoRA' not in results_dict:
        print("  - LoRA 모델 결과가 없어 예시 생성을 건너뜁니다.")
        return []
    lora_results = results_dict['LoRA']
    silenced_models = [k for k in results_dict.keys() if 'Silenced' in k]

    quality_drops = []
    for i, lora_result in enumerate(lora_results):
        for model_name in silenced_models:
            silenced_result = results_dict[model_name][i]
            bleu_drop = lora_result['performance']['bleu'] - silenced_result['performance']['bleu']
            if bleu_drop > 0.1:  # 유의미한 하락만 추출
                quality_drops.append({
                    'question_idx': i,
                    'question': lora_result['question'],
                    'reference': lora_result['reference_answer'],
                    'lora_answer': lora_result['generated_answer'],
                    'lora_bleu': lora_result['performance']['bleu'],
                    'silenced_model': model_name,
                    'silenced_answer': silenced_result['generated_answer'],
                    'silenced_bleu': silenced_result['performance']['bleu'],
                    'bleu_drop': bleu_drop
                })

    if not quality_drops:
        print("  - 유의미한 품질 하락 예시를 찾지 못했습니다.")
        return []

    quality_drops.sort(key=lambda x: x['bleu_drop'], reverse=True)
    top_examples = quality_drops[:10]

    examples_df = pd.DataFrame(top_examples)
    examples_df.to_csv(os.path.join(OUTPUT_DIR, 'answer_quality_examples.csv'),
                       index=False, encoding='utf-8-sig')
    print("  - 답변 품질 변화 예시 CSV 저장 완료")
    return top_examples


def save_all_results(results_dict, activations_dict, individual_neurons, simultaneous_neurons):
    print("\n💾 모든 결과 종합 및 저장 시작...")
    summary_stats = {"metadata": {
        "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "models_compared": list(results_dict.keys()),
        "silencing_experiments": {
            "individual_neurons": individual_neurons,
            "simultaneous_neurons": simultaneous_neurons
        },
        "total_questions": len(list(results_dict.values())[0])
    }}
    for model_name, results in results_dict.items():
        df = pd.DataFrame([r['performance'] for r in results])
        summary_stats[model_name] = {
            "average_scores": df.mean().to_dict(),
            "std_dev_scores": df.std().to_dict()
        }

    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w', encoding='utf-8') as f:
        # [수정] cls=NumpyEncoder 옵션을 추가하여 커스텀 번역기 사용
        json.dump(summary_stats, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    print("  - 통계 요약 JSON 저장 완료.")

    try:
        flat_data = []
        num_questions = len(list(results_dict.values())[0])
        model_names = list(results_dict.keys())
        for i in range(num_questions):
            row = {'question_idx': i}
            base_res = results_dict[model_names[0]][i]
            row['question'] = base_res['question']
            row['reference_answer'] = base_res['reference_answer']
            for model_name in model_names:
                res = results_dict[model_name][i]
                row[f'{model_name}_answer'] = res['generated_answer']
                for metric, value in res['performance'].items():
                    row[f'{model_name}_{metric}'] = value
                top_neurons_str = ", ".join(
                    [f"#{n['neuron_idx']}({n['activation']:.2f})" for n in res['top5_activated_neurons']])
                row[f'{model_name}_top5_neurons'] = top_neurons_str
            flat_data.append(row)
        detailed_df = pd.DataFrame(flat_data)
        cols_to_front = ['question_idx', 'question', 'reference_answer']
        cols = cols_to_front + [col for col in detailed_df.columns if col not in cols_to_front]
        detailed_df = detailed_df[cols]
        detailed_df.to_csv(os.path.join(OUTPUT_DIR, 'detailed_comparison_by_question.csv'), index=False,
                           encoding='utf-8-sig')
        print("  - 질문별 상세 비교 CSV 저장 완료.")
    except Exception as e:
        print(f"⚠️ CSV 저장 오류: {e}")


def main():
    print("=" * 60)
    print("🔬 BWR 특화 모델 종합 분석 스크립트 (최종 정리 버전)")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        qa_pairs = load_eval_data()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ 초기화 오류: {e}");
        return

    try:
        # 1. Base 및 LoRA 모델 분석 실행
        print("\n⏳ Base 모델 로드 중...")
        base_model = Gemma3ForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
        base_activations, base_results = analyze_model(base_model, tokenizer, qa_pairs, "Base")
        del base_model;
        cleanup_gpu()

        print("\n⏳ LoRA 모델 로드 및 병합 중...")
        base_model_for_lora = Gemma3ForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                                                device_map="auto")
        lora_model = PeftModel.from_pretrained(base_model_for_lora, LORA_MODEL_PATH)
        lora_model_merged = lora_model.merge_and_unload()
        del base_model_for_lora, lora_model;
        cleanup_gpu()
        print("✅ LoRA 모델 로드 및 병합 완료")

        lora_activations, lora_results = analyze_model(lora_model_merged, tokenizer, qa_pairs, "LoRA")

        # 2. 분석 대상 뉴런 선정
        print("\n⚡️ 상위 뉴런 탐색 및 분석 대상 선정...")
        activation_diff = np.mean(lora_activations, axis=0) - np.mean(base_activations, axis=0)

        # 2-1. 모든 뉴런을 변화량의 절댓값 크기 순으로 정렬
        all_sorted_indices = np.argsort(np.abs(activation_diff))[::-1]

        # 2-2. 전체 양상을 시각화할 대상 선정 (상위 12개)
        indices_for_visualization = all_sorted_indices[:12]

        # 2-3. 집중 분석(사일런싱)할 대상 선정 (증폭 5개 + 억제 1개)
        activated_neurons = [idx for idx in all_sorted_indices if activation_diff[idx] > 0]
        suppressed_neurons = [idx for idx in all_sorted_indices if activation_diff[idx] < 0]

        top_activated = activated_neurons[:NUM_TOP_ACTIVATED_TO_SILENCE]
        top_suppressed = suppressed_neurons[:NUM_TOP_SUPPRESSED_TO_SILENCE]

        indices_for_silencing = top_activated + top_suppressed

        if not indices_for_silencing:
            print("⚠️ 분석할 뉴런을 찾지 못했습니다. 분석을 종료합니다.");
            return

        print(f"✅ 전체 시각화 대상 뉴런 ({len(indices_for_visualization)}개): {indices_for_visualization.tolist()}")
        print(f"🎯 집중 분석(사일런싱) 대상 뉴런 ({len(indices_for_silencing)}개): {indices_for_silencing}")

        # 3. 사일런싱 실험 진행
        results_dict = {"Base": base_results, "LoRA": lora_results}
        last_mlp_layer = get_last_mlp_layer(lora_model_merged)

        for neuron_idx in indices_for_silencing:
            model_name = f"LoRA-Silenced-#{neuron_idx}"
            silence_hook = SilenceHook(int(neuron_idx))
            hook_handle = last_mlp_layer.register_forward_hook(silence_hook)
            _, silenced_results = analyze_model(lora_model_merged, tokenizer, qa_pairs, model_name)
            results_dict[model_name] = silenced_results
            hook_handle.remove()
            print(f"🤫 뉴런 #{neuron_idx}에 대한 SilenceHook 제거 완료.")

        model_name = f"LoRA-Silenced-Key{len(indices_for_silencing)}"
        silence_hook = SilenceHook(indices_for_silencing)
        hook_handle = last_mlp_layer.register_forward_hook(silence_hook)
        _, silenced_results = analyze_model(lora_model_merged, tokenizer, qa_pairs, model_name)
        results_dict[model_name] = silenced_results
        hook_handle.remove()
        print(f"🤫 뉴런 {indices_for_silencing}에 대한 동시 SilenceHook 제거 완료.")

        # 4. 결과 시각화 및 저장
        create_paper_visualizations(base_activations, lora_activations, indices_for_visualization)
        create_silencing_visualizations(results_dict)
        create_statistical_analysis(results_dict)
        create_question_type_analysis(results_dict)
        create_answer_quality_examples(results_dict)
        save_all_results(results_dict, None,
                         individual_neurons=indices_for_silencing,
                         simultaneous_neurons=indices_for_silencing)

        print("\n🎉 모든 분석이 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"❌ 스크립트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()
