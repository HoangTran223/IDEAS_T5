# Training

### For my model
For GPT2-base, run:
```bash
bash scripts/gpt2/multicost_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/multicost_tinyllama.sh
```

### SFT for teacher models
For Qwen1.5-1.8B (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_teacher_qwen.sh
```

- Đã có sẵn một model SFT của Qwen1.5-1.8B (full fine-tuning):

https://drive.google.com/drive/folders/11Eba3lgnWZGjFW2EPUQVC1nLv6-LRI-9?usp=sharing

For LLaMA2-7B (LoRA), run:
```bash
bash scripts/tinyllama/sft_teacher_llama2.sh
```

For Mistral-7B (LoRA), run:
```bash
bash scripts/tinyllama/sft_teacher_mistral.sh
```

### SFT for student models
For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_gpt2_base.sh
```

For TinyLLaMA-1.1B (LoRA), run:
```bash
bash scripts/tinyllama/sft_tinyllama.sh
```

P.S. You may encounter an error **when directly loading the model checkpoint of TinyLLaMA**. This is because of the mismatched versions of `transformers` between TinyLLaMA suggested (4.31) and the one you use.
A concise solution to fix this can be referred to in [this issue](https://github.com/songmzhang/DSKD/issues/8).


### Baseline: Dual-Space KD with CMA
For GPT2-base, run:
```bash
bash scripts/gpt2/dskd_cma_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/dskd_cma_tinyllama.sh
```

### Baseline: Logits Alignment by Minimum Edit Distance 
For GPT2-base, run:
```bash
bash scripts/gpt2/minedit_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/minedit_tinyllama.sh
```

### Baseline: Universal Logit Distillation 
For GPT2-base, run:
```bash
bash scripts/gpt2/uld_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/uld_tinyllama.sh
```


### Baseline: Dual-Space KD framework
For GPT2-base, run:
```bash
bash scripts/gpt2/dskd_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/dskd_tinyllama.sh
```

Also, you can change the distance functions using `KD_OBJ` in the above scripts.


### Baseline: Vanilla KD framework
For GPT2-base, run:
```bash
bash scripts/gpt2/vanilla_kd_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/vanilla_kd_tinyllama.sh
```

You can change the distance functions (e.g., KL Divergence, Reverse KL Divergence, JS Divergence, etc.) using `KD_OBJ` in the above scripts.



### File Structures in Output Directory
The output directory will be created under `./outputs` automatically after you run the training scripts. 
For full fine-tuning, the file structure of the output directory is as follows (take gpt2 SFT as an example):
```
./outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```
For LoRA fine-tuning, the file structure of the output directory is as follows (take TinyLLaMA LoRA SFT as an example):
```
./outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```



## Evaluation
### Evaluate Full Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```
According to the above structure, `CKPT_PATH` is the **absolute path** of the model files like `/home/xxx/DSKD/outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../epochA_step...`.

### Evaluate LoRA Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```
Please note that `MODEL_PATH` in `run_eval_lora.sh` should be changed for different base models (TinyLLaMA, LLaMA2, Mistral).

Similarly, `LORA_ADAPTER_PATH` is the **absolute path** of the LoRA adapter files like `/home/xxx/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../epochA_step...`.

## Deepspeed
- Cần Deepspeed>= 0.14.0 (Repo github đã có thư mục deepspeed rồi nên không cần tải thư viện này)

## Data
The processed data used in our paper can be downloaded [here](https://drive.google.com/drive/folders/1ZUsNVgWevACV9D-AHVNi9C7PX_2itzb8?usp=sharing).

## Models
You can download the corresponding model files (e.g., `pytorch_model.bin` or `model.safetensors`) of LLMs used in this paper into `model_hub/*/*/`.

Here are the links of these models on huggingface:
- GPT2-120M: [Here](https://huggingface.co/openai-community/gpt2)
- GPT2-1.5B (trained on Dolly by Gu et al.): [Here](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- Qwen1.5-1.8B: [Here](https://huggingface.co/Qwen/Qwen1.5-1.8B)
- TinyLLaMA-1.1B: [Here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Llama2-7B: [Here](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Mistral-7B: [Here](https://huggingface.co/mistralai/Mistral-7B-v0.1)

