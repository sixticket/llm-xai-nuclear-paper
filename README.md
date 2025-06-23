# llm-xai-nuclear-paper

# Mechanistic Interpretability of a Domain-Adapted Large Language Model for Nuclear Engineering Applications

This repository contains the source code, data, and analysis scripts for the paper: "Mechanistic Interpretability of a Domain-Adapted Large Language Model for Nuclear Engineering Applications".

## Abstract

The integration of Large Language Models (LLMs) into safety-critical domains such as nuclear engineering necessitates a deep understanding of their internal reasoning processes. This paper presents a novel methodology for interpreting how an LLM encodes and utilizes domain-specific knowledge, using a Boiling Water Reactor (BWR) system as a case study. We adapted a general-purpose LLM (Gemma-3-1b-it) to the nuclear domain using a parameter-efficient fine-tuning technique known as Low-Rank Adaptation (LoRA). By comparing the neuron activation patterns of the base model with the fine-tuned model, we identified a sparse set of neurons whose behavior was significantly altered during the adaptation process. To probe the causal role of these "specialized" neurons, we employed a neuron silencing technique. Our results demonstrate that deactivating a small, targeted group of these neurons led to a statistically significant degradation in task performance. This work provides a concrete methodology for tracing the mechanistic underpinnings of domain expertise within an LLM, offering a critical step toward building more transparent, reliable, and verifiable AI systems for nuclear science and engineering.

## Repository Structure

```
├── main.tex                           # Main LaTeX source file for the paper
├── references.bib                     # BibTeX file for managing all references
├── data/
│   └── bwr_eval_stratified.json      # Question-Answering dataset in JSON format
├── figures/                          # All figures and plots used in the paper
└── scripts/                          # Python scripts used in this study
    ├── extract_qa_gpt4o.py           # Script to generate QA dataset from source documents
    ├── LoRA_fine_tune.py             # Script for fine-tuning the base model with QLoRA
    └── last_analysis.py              # Script for neuron activation analysis and silencing experiments
```

## How to Reproduce

### 1. Dependencies

It is recommended to use a Python virtual environment. Install the required libraries using pip:

```bash
pip install torch transformers peft accelerate bitsandbytes scipy matplotlib numpy openai pypdf
```

### 2. Running the Experiments

The scripts should be run in the following order:

1. **Generate Dataset:** Run `scripts/extract_qa_gpt4o.py` to create the `bwr_eval_stratified.json` file. 
   > **Note:** This requires access to the original source PDFs and a valid OpenAI API key.

2. **Fine-tune Model:** Run `scripts/LoRA_fine_tune.py` to train the LoRA model using the generated dataset.

3. **Analyze Results:** Run `scripts/last_analysis.py` to perform the neuron analysis, generate all figures for the paper, and save the quantitative results.

### 3. Compiling the Paper

To compile the LaTeX paper (`main.tex`) and generate a PDF with all references correctly linked, use `latexmk`:

```bash
latexmk -pdf main.tex
```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{lee2025mechanistic,
  title={Mechanistic Interpretability of a Domain-Adapted Large Language Model for Nuclear Engineering Applications},
  author={Lee, Yoon Pyo},
  journal={Nuclear Science and Engineering (Submitted)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues related to this work, please contact:
- **Author:** Yoon Pyo Lee
- **Email:** [lukeyounpyo@hanyang.ac.kr]

## Acknowledgments

We acknowledge the contributions of the nuclear engineering community and the open-source machine learning community for making this research possible.
