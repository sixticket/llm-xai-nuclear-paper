import torch
import json
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import os
from datetime import datetime
import time

# 설정
MODEL_PATH = r"path/to/gemma-3-1b-it"
DATA_PATH = r"path/to/qa/dataset"
OUTPUT_DIR = r"path/to/analysis/results"

print("🚀 BWR 전문 Gemma 1B LoRA 파인튜닝 시작!")
print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# 진행도 표시 콜백
class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            current_step = state.global_step
            total_steps = state.max_steps
            elapsed = time.time() - self.start_time
            progress = (current_step / total_steps) * 100
            eta = (elapsed / current_step) * (total_steps - current_step)

            # 진행 바
            bar_length = 25
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # 현재 epoch 계산
            current_epoch = current_step / (total_steps / args.num_train_epochs)

            if current_step % 10 == 0:  # 10스텝마다 출력
                print(f"\n🔄 [{bar}] {progress:.1f}% | "
                      f"Epoch: {current_epoch:.2f}/{args.num_train_epochs} | "
                      f"Step: {current_step}/{total_steps} | "
                      f"Loss: {logs.get('train_loss', 0):.4f} | "
                      f"경과: {elapsed / 60:.0f}분 | 남은시간: {eta / 60:.0f}분")


# 1. Stratified 데이터 로드
def load_stratified_data():
    """문서별로 미리 분할된 train/eval 데이터 로드"""

    train_filepath = os.path.join(DATA_PATH, "bwr_train_stratified.json")
    eval_filepath = os.path.join(DATA_PATH, "bwr_eval_stratified.json")

    # Train 데이터 로드
    with open(train_filepath, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        train_qa_pairs = train_data['qa_pairs']

    # Eval 데이터 로드
    with open(eval_filepath, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        eval_qa_pairs = eval_data['qa_pairs']

    print(f"📄 Train 데이터: {len(train_qa_pairs)}개 로드")
    print(f"📄 Eval 데이터: {len(eval_qa_pairs)}개 로드")

    # 문서별 분포 확인
    train_sources = {}
    eval_sources = {}

    for qa in train_qa_pairs:
        source = qa.get('source', 'Unknown')
        train_sources[source] = train_sources.get(source, 0) + 1

    for qa in eval_qa_pairs:
        source = qa.get('source', 'Unknown')
        eval_sources[source] = eval_sources.get(source, 0) + 1

    print("📊 Train 세트 문서별 분포:")
    for source, count in train_sources.items():
        print(f"  {source}: {count}개")

    print("📊 Eval 세트 문서별 분포:")
    for source, count in eval_sources.items():
        print(f"  {source}: {count}개")

    return train_qa_pairs, eval_qa_pairs


# 2. 데이터 전처리
def preprocess_data(train_qa_pairs, eval_qa_pairs, tokenizer):
    print("🔄 데이터 전처리 중...")

    # Train 데이터 전처리
    train_texts = []
    for qa in train_qa_pairs:
        text = f"<bos><start_of_turn>user\n{qa['question']}<end_of_turn>\n<start_of_turn>model\n{qa['answer']}<end_of_turn><eos>"
        train_texts.append(text)

    train_tokenized = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    train_tokenized["labels"] = train_tokenized["input_ids"].clone()

    train_dataset = Dataset.from_dict({
        "input_ids": train_tokenized["input_ids"],
        "attention_mask": train_tokenized["attention_mask"],
        "labels": train_tokenized["labels"]
    })

    # Eval 데이터 전처리
    eval_texts = []
    for qa in eval_qa_pairs:
        text = f"<bos><start_of_turn>user\n{qa['question']}<end_of_turn>\n<start_of_turn>model\n{qa['answer']}<end_of_turn><eos>"
        eval_texts.append(text)

    eval_tokenized = tokenizer(
        eval_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    eval_tokenized["labels"] = eval_tokenized["input_ids"].clone()

    eval_dataset = Dataset.from_dict({
        "input_ids": eval_tokenized["input_ids"],
        "attention_mask": eval_tokenized["attention_mask"],
        "labels": eval_tokenized["labels"]
    })

    print(f"✅ 학습 데이터: {len(train_dataset)}개")
    print(f"✅ 테스트 데이터: {len(eval_dataset)}개")

    return {"train": train_dataset, "test": eval_dataset}


def main():
    print(f"🔧 CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"🔧 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    print("📥 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma3ForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        use_cache=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"📊 모델 파라미터: {model.num_parameters():,}")
    print(f"📊 GPU 메모리: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f}GB")

    # Stratified 데이터 로드
    train_qa_pairs, eval_qa_pairs = load_stratified_data()
    dataset = preprocess_data(train_qa_pairs, eval_qa_pairs, tokenizer)

    model.train()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("🔧 LoRA 설정 적용 중...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 학습 가능 파라미터: {trainable_params:,} ({trainable_params / all_params * 100:.2f}%)")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=2e-5,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_num_workers=0,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        remove_unused_columns=False,
    )

    total_steps = (len(dataset["train"]) // (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
    print(f"📊 총 학습 스텝: {total_steps}")
    print(f"💡 예상 소요 시간: {total_steps * 3 / 60:.0f}분")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()]
    )

    print("\n🚀 LoRA 파인튜닝 시작!")
    print("📊 실시간 진행도가 표시됩니다...\n")

    try:
        trainer.train()

        print("\n💾 모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)

        print(f"\n✅ 학습 완료! 모델 저장: {OUTPUT_DIR}")
        print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 테스트
        print("\n🧪 학습된 모델 테스트:")
        model.eval()

        test_prompts = [
            "What is a boiling water reactor?",
            "How do control rods work in BWR?"
        ]

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # 50→100으로 증가
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"\nQ: {prompt}")
            print(f"A: {response}")

    except Exception as e:
        print(f"\n❌ 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
