"""LoRA-tune Qwen2.5-1.5B-Instruct on the synthesized SFT set.

Targets the local model's measured failure, not general quality: the eval
showed persona and guardrails mostly hold at 7B but tools never get called and
missing facts get invented (Finding 04). The training mix is therefore mostly
tool-call traces and in-character refusals, rendered through the model's own
chat template with the tool schemas attached — the same shape the agent loop
produces at inference time.

1.5B in bf16 fits an 8 GB laptop GPU without quantization (LoRA, batch 1,
grad-accum, gradient checkpointing). QLoRA is for the 7B attempt, if any.

Run inside finetune/.venv:
    python train_lora.py
    python train_lora.py --smoke   # 8 samples, 1 epoch, proves the plumbing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def to_template_messages(sample: dict) -> tuple[list[dict], list[dict] | None]:
    """Our on-disk trace -> the shape Qwen's chat template expects.

    Tool calls become {"function": {"name", "arguments"}} entries; tool results
    keep role "tool". The system/user/assistant lines pass through untouched.
    """
    rendered: list[dict] = []
    for message in sample["messages"]:
        if "tool_calls" in message:
            rendered.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        else:
            rendered.append(message)
    tools = sample.get("tools") or None
    return rendered, tools


def load_rows(tokenizer, path: Path) -> Dataset:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        messages, tools = to_template_messages(sample)
        texts.append(
            tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        )
    return Dataset.from_dict({"text": texts})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    # v1 ran 3 epochs at 1e-4 on 172 samples: loss 0.5, token-accuracy 0.90,
    # and an eval that said "memorized the refusals, forgot how to answer".
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--out", default=str(HERE / "out" / "lora-qwen2.5-1.5b"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = load_rows(tokenizer, HERE / "data" / "sft_train.jsonl")
    if args.smoke:
        dataset = dataset.select(range(8))

    lengths = [len(tokenizer(t).input_ids) for t in dataset["text"]]
    print(f"{len(dataset)} samples · token length p50={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=1.0 if args.smoke else args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_length=2048,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
