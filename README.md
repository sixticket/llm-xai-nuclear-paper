# Mechanistic Interpretability of Domain-Adapted Large Language Models for Nuclear Engineering Applications

This repository contains the code and methodology for the paper "Mechanistic Interpretability of a Domain-Adapted Large Language Model for Nuclear Engineering Applications" submitted to Nuclear Technology.

## Overview

This project demonstrates a novel methodology for interpreting how Large Language Models (LLMs) encode and utilize domain-specific knowledge in safety-critical nuclear engineering applications. We adapt a general-purpose LLM (Gemma-3-1b-it) to the nuclear domain using LoRA fine-tuning and employ neuron silencing techniques to identify specialized neural circuits responsible for domain expertise.

## Key Contributions

- **Glass Box AI**: Transform opaque "black box" models into transparent "glass box" systems for nuclear safety applications
- **Mechanistic Analysis**: Identify specific neurons whose activation patterns change during domain adaptation
- **Causal Intervention**: Use neuron silencing to test the functional importance of specialized neurons
- **Nuclear-Grade Assurance**: Provide a pathway toward AI verification and validation for nuclear regulatory compliance

## Repository Structure

```
├── README.md
├── requirements.txt
├── extract_qa_gpt4o.py      # QA dataset generation using GPT-4o
├── LoRA_fine_tune.py        # LoRA fine-tuning implementation
└── last_analysis.py         # Neuron analysis and silencing experiments
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+

### Setup

```bash
git clone https://github.com/[username]/nuclear-llm-interpretability
cd nuclear-llm-interpretability
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install torch transformers peft datasets evaluate
pip install numpy pandas matplotlib scipy tqdm
pip install openai  # For QA generation (optional)
```

## Usage

### 1. Data Generation (Optional)

Generate domain-specific QA pairs using GPT-4o:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Generate QA dataset
python extract_qa_gpt4o.py
```

**Note**: You need to provide your own nuclear engineering documents as source material.

### 2. LoRA Fine-tuning

Train the domain-adapted model:

```bash
python LoRA_fine_tune.py
```

### 3. Mechanistic Analysis

Perform neuron activation analysis and silencing experiments:

```bash
python last_analysis.py
```

## Configuration

Update the following paths in each script according to your setup:

```python
# In each .py file, modify these variables:
BASE_MODEL_PATH = "path/to/gemma-3-1b-it"
LORA_MODEL_PATH = "path/to/output/lora/model"
DATA_PATH = "path/to/qa/dataset"
OUTPUT_DIR = "path/to/analysis/results"
```

### LoRA Configuration
Key parameters in `LoRA_fine_tune.py`:

```python
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Alpha
    target_modules=[...],   # Target attention/MLP layers
    lora_dropout=0.05,      # Dropout rate
)
```

### Analysis Parameters
Configure neuron analysis in `last_analysis.py`:

```python
NUM_TOP_ACTIVATED_TO_SILENCE = 5    # Amplified neurons to analyze
NUM_TOP_SUPPRESSED_TO_SILENCE = 1   # Suppressed neurons to analyze
```

## Results

The analysis generates:

- **Neuron activation visualizations**: Changes in activation patterns
- **Performance metrics**: BLEU/ROUGE scores before and after silencing
- **Statistical significance tests**: Wilcoxon signed-rank tests
- **Quality degradation examples**: Concrete examples of performance loss

## Data

Due to the sensitive nature of nuclear domain data, the specific QA dataset used in the paper is not included in this repository. However, the complete data generation methodology is provided in `extract_qa_gpt4o.py`.

Users can create their own domain-specific datasets by:

1. Collecting authoritative technical documents (IAEA, NRC, etc.)
2. Using the provided GPT-4o extraction script
3. Following the stratified sampling approach described in the paper

## Nuclear Safety Considerations

This methodology is designed with nuclear safety principles in mind:

- **Plant-specific deployment**: Each facility maintains isolated AI systems
- **Verification & Validation**: Traceable neural circuits for regulatory compliance
- **Defense-in-Depth**: Multiple layers of interpretability analysis
- **ALARA principle**: Minimizing AI-related risks through transparency

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{lee2024mechanistic,
  title={Mechanistic Interpretability of a Domain-Adapted Large Language Model for Nuclear Engineering Applications},
  author={Lee, Yoon Pyo},
  journal={Nuclear Technology},
  year={2024},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Gemma team at Google for the base model
- Hugging Face for the transformers and PEFT libraries

## Contact

For questions about the methodology or implementation:

- **Author**: Yoon Pyo Lee
- **Email**: lukeyounpyo@hanyang.ac.kr
- **Institution**: Department of Nuclear Engineering, Hanyang University

## Disclaimer

This research is for academic purposes only. Any deployment in actual nuclear facilities must undergo proper regulatory review and approval processes in accordance with relevant nuclear safety standards (10 CFR 50 Appendix B, IEEE Std 7-4.3.2, etc.).
