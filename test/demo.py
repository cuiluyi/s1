# 环境安装（确保版本兼容性）
# pip install transformers datasets trl peft accelerate bitsandbytes

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. 加载模型和分词器
model_path = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 2. 冻结除lm_head外的所有参数
for name, param in model.named_parameters():
    if "lm_head" not in name:
        param.requires_grad = False

# 验证参数冻结情况
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# 3. 准备数据集
dataset = load_dataset("simplescaling/s1K", split="train")

# 定义数据格式化函数（根据模型需求调整）
def formatting_func(example):
    text = f"[INST] {example['input']} [/INST] {example['output']}</s>"
    return {"text": text}

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./qwen_math_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    max_grad_norm=0.3
)

# 5. 初始化Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    max_seq_length=1024,
    dataset_text_field="text",
)

# 6. 开始训练
trainer.train()

# 7. 保存结果
trainer.save_model("./qwen_math_finetuned")
tokenizer.save_pretrained("./qwen_math_finetuned")