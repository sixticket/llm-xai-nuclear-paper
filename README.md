# Mechanistic Interpretability of LoRA-Adapted LLMs for Nuclear Reactor Safety

This repository contains the implementation code for the paper "Mechanistic Interpretability of LoRA-Adapted Language Models for Nuclear Reactor Safety Applications" submitted to Nuclear Technology.

## Overview

This work addresses the critical challenge of deploying Large Language Models (LLMs) in safety-critical nuclear engineering applications by developing a methodology to interpret and verify the internal mechanisms of domain-adapted models.

### Key Contributions
- 🔬 First application of mechanistic interpretability to nuclear safety domain
- 🧠 Identification of specialized neurons encoding BWR technical knowledge
- 📊 Causal verification through neuron silencing experiments
- 🏭 Pathway toward regulatory-compliant AI in nuclear operations

## Repository Structure

```
├── extract_qa_gpt4o.py      # Dataset creation from BWR technical documents
├── LoRA_fine_tune.py        # Domain adaptation using LoRA
├── last_analysis.py         # Neuron analysis and silencing experiments
└── README.md
```

## Requirements

```bash
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets
evaluate
numpy
pandas
matplotlib
scipy
openai  # for dataset creation
```

## Quick Start

### 1. Dataset Preparation

Generate Q&A pairs from BWR technical documents:

```bash
python extract_qa_gpt4o.py
```

This creates domain-specific training data from:
- IAEA BWR Simulator documentation
- OECD/NEA Severe Accident Mitigation Guidelines
- NRC Technical Training Center materials

### 2. LoRA Fine-tuning

Adapt the base Gemma-3-1b-it model to nuclear domain:

```bash
python LoRA_fine_tune.py
```

Key parameters:
- LoRA rank: 8
- Target modules: All attention and MLP projections
- Training epochs: 2
- Learning rate: 2e-5

### 3. Mechanistic Analysis

Identify and analyze specialized neurons:

```bash
python last_analysis.py
```

This script:
- Compares neuron activations between base and fine-tuned models
- Identifies top-changed neurons (amplified/suppressed)
- Performs causal intervention through neuron silencing
- Generates all figures and statistical analyses

## Key Results

### Neuron Specialization
- **Amplified neurons** (e.g., #1066, #1086): Encode domain-specific technical concepts
- **Suppressed neurons** (e.g., #941): Reduce general language tendencies
- **Circuit behavior**: Collective silencing causes significant performance degradation

### Performance Metrics
- Base model BLEU: 0.027
- LoRA model BLEU: 0.150 
- LoRA-Silenced-Key6 BLEU: 0.139 (statistically significant drop)

## Reproducing Paper Results

1. **Figure 1**: Top neuron activation changes
   - Generated automatically by `last_analysis.py`
   - Shows amplified (red) and suppressed (blue) neurons

2. **Figure 2-3**: Detailed activation distributions
   - Boxplot comparisons for top 12 neurons
   - Demonstrates systematic activation shifts

3. **Figure 4**: Statistical significance analysis
   - Wilcoxon signed-rank test results
   - Shows p-values for performance differences

4. **Table II**: Answer quality examples
   - Saved as `answer_quality_examples.csv`
   - Demonstrates safety-critical failure modes

## Data Availability

Due to the sensitive nature of nuclear safety documentation, the raw training texts are not included. However:
- The Q&A dataset generation process is fully reproducible
- Source documents are publicly available from IAEA, OECD/NEA, and NRC
- Generated datasets can be shared upon request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lee2025mechanistic,
  title={Mechanistic Interpretability of LoRA-Adapted Language Models for Nuclear Reactor Safety Applications},
  author={Lee, Yoon Pyo},
  journal={Nuclear Technology},
  year={2025},
  note={Submitted}
}
```

## Safety Notice

This research is intended for academic purposes and regulatory framework development. Any deployment in actual nuclear facilities must undergo rigorous verification and validation according to applicable regulations (e.g., 10 CFR 50 Appendix B, IEEE Std 7-4.3.2).

## Contact

Yoon Pyo Lee  
Department of Nuclear Engineering  
Hanyang University  
Email: lukeyounpyo@hanyang.ac.kr

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

This work was conducted as an undergraduate research project at Hanyang University. Special thanks to the nuclear engineering community for their commitment to safety and innovation.
