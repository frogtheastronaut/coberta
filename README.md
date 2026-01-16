---
language: 
- en
license: mit
tags:
- roberta
- mlm
- small-model
- academic-language
- synthetic-data
- resource-efficient
- consumer-hardware
pipeline_tag: fill-mask
datasets:
- HuggingFaceTB/cosmopedia
library_name: transformers
---
# CoBERTa: 24M Parameter Academic Language Model

<div align="center">
  <img src="https://img.shields.io/badge/Parameters-24.5M-blue" alt="24.5M Parameters">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</div>

## Model Description

CoBERTa is a **24.5 million parameter** RoBERTa-style masked language model specifically trained on synthetic academic data. It demonstrates that **high-quality synthetic data** can compensate for model scale, enabling domain specialization on consumer hardware.

It achieves academic language proficiency with 5× fewer parameters than comparable models, trained in around 6 hours on a MacBook Pro.

### Model Architecture
- **Type**: Encoder-only transformer
- **Layers**: 12
- **Hidden Size**: 192
- **Attention Heads**: 6
- **Parameters**: 24,506,224
- **Vocabulary**: 35,000 tokens
- **Maximum Sequence Length**: 512 tokens

## Training Details

### Training Data
- **Source**: 50,000 filtered samples from [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- **License**: MIT

### Training Procedure
- **Framework**: Apple MLX
- **Hardware**: MacBook Pro M4 (16GB unified memory)
- **Training Time**: 6 hours
- **Batch Size**: 32
- **Learning Rate**: 9e-4 with linear warmup
- **Objective**: Masked language modeling (15% token masking)

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation you should look at a model like GPT2.

### Basic Usage
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
model = AutoModelForMaskedLM.from_pretrained("appleroll/coberta-base")
tokenizer = AutoTokenizer.from_pretrained("appleroll/coberta-base")
text = "The key to effective communication is to [MASK] clearly and listen actively."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits
# Get top predictions for [MASK]
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
mask_token_logits = predictions[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(f"{tokenizer.decode([token])}")
```

### Citation

If you use CoBERTa in your research, please cite:
```bibtex
@misc{coberta,
    title     = {CoBERTa: Training Domain-Expert Language Models on Consumer Hardware},
    url       = {https://huggingface.co/appleroll/coberta-base},
    author    = {Zhang, Ethan},
    year      = {2025}
}
```

### Additional Information
Authors:
- Ethan Zhang (Independent Researcher)

### Contact

For questions, feedback, or collaboration: ethanzhangyixuan@gmail.com

### Version

Current: v0.5 (Experimental Release)

Previous: None

### License

This model is licensed under the MIT License. See the LICENSE file for details.