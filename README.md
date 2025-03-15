# AI and Defense Strategy: Text Analysis

[![DOI](https://zenodo.org/badge/621097589.svg)](https://zenodo.org/badge/latestdoi/621097589)

![coverage badge](https://raw.githubusercontent.com/ajkeith/StrategyDocumentAnalysis/main/coverage.svg)

This python project analyzes national AI and defense strategy documents using zero-shot text classification. The project focuses on Southeast Asia and nearby countries, specifically: Australia, Indonesia, Malaysia, Singapore, Thailand, and Vietnam. 

## Getting Started

```python
python -m main.py
```

## Usage

```python
import os
from textanalysis import analysis

path = os.path.join(os.getcwd(), 'data', 'policies', 'australia_defense.pdf')
temp = analysis.extract_pdfs(path)
df, fig = analysis.analyze_corpus(temp)
```

The result of `analyze_corpus` is a dataframe of classified text (by topic and sentiment) and an interactive plot of the topic and sentiment by text chunk. 

## Algorithm Details

This code uses the `facebook/bart-large-mnli` large BART model from [Hugging Face](https://huggingface.co/facebook/bart-large-mnli). This is a [MutliNLI](https://huggingface.co/datasets/multi_nli)-tuned model based on [BART](https://arxiv.org/abs/1910.13461) and used here for zero-shot text classification. 

This code also uses the `distilbert-base-uncased-finetuned-sst-2-english` model from [Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). This is a fine-tuned model based on [DistilBERT](https://arxiv.org/abs/1910.01108) and used here for sentiment classification. 

`distilbert-base-uncased-finetuned-sst-2-english` has [strong evaluation results](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) in terms of accuracy and precision:

<img src="https://user-images.githubusercontent.com/26749415/230005604-98fd3980-7d26-48b9-92f8-82197664a339.png" width="500">

However, it is also subject to [risks, limitations, and biases](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english#risks-limitations-and-biases). 

## Data

The national-level AI strategies or policies for GPAI and each country under consideration are included as `.pdf`s in the `data/policies` [directory](https://github.com/ajkeith/StrategyDocumentAnalysis/tree/main/data/policies). The text-only version of those policies are included as `.txt`s in the `data/texts` [directory](https://github.com/ajkeith/StrategyDocumentAnalysis/tree/main/data/texts).

The membership assessment [metrics](https://gpai.ai/about/membership-and-observers-metrics.pdf) for the Global Partnership on Artificial Intelligence (GPAI) are included in the `data/metrics` [directory](https://github.com/ajkeith/StrategyDocumentAnalysis/tree/main/data/metrics). This directory includes the source documents and consolidated metrics for the countries under consideration. The metrics are defined in the 2021 GPAI _Frame for letter of intent and reference metrics to support the assessment of GPAI Membership_ (also available in the same directory). The datasets are organized with the following identifiers:

| Identifier | Dataset |
| --- | --- |
| aidv | AI and Democratic Values Index |
| aigs | AI Global Surveillance Index |
| aii | Stanford AI Index |
| cri | Commitment to Reducing Inequality Index |
| di | Democracy Index |
| gai | Global AI Index |
| gair | Government AI Readiness Index |
| gfs | Global Freedom Score |
| libdem | V-Dem Liberal Democracy Index |
| odi | Open Data Index |
| ttaip | Total number of 10% top-cited AI scientific publications, fractional counts ([source](https://oecd.ai/en/oecd-metrics-and-methods?selectedArea=bibliometrics&selectedVisualization=total-number-of-10-top-cited-ai-scientific-publications-fractional-counts)) |

Intermediate data files and output figures and tables are included in the `data/output` [directory](https://github.com/ajkeith/StrategyDocumentAnalysis/tree/main/data/output).

## Results

![image](https://github.com/user-attachments/assets/7944af29-5ed8-4b6a-a9c2-0cea6048c1e4)

![image](https://github.com/user-attachments/assets/d5a9e666-f8de-40dd-856d-e75a79c7b31f)

![image](https://github.com/user-attachments/assets/d9f76400-9cff-4e92-9164-7f36e7f8e901)

![image](https://github.com/user-attachments/assets/f44cf649-3629-48c8-a018-cf9db0c3b7b6)

Additionally, the following figure shows the sentiment and topic classficiation through Singapore's National AI Strategy.

![text classificaiton figure](https://user-images.githubusercontent.com/26749415/230006657-b511a380-7872-41ac-bd6b-daf230fb9790.png)

## Policy Implications
 - Global policy on artificial intelligence could increase acceptance in Southeast Asia by emphasising human capital promotion to align with priorities in Southeast Asian policies.
 - The ethical content of global policy on artificial intelligence is generally aligned with the ethical content of Southeast Asian policies on artificial intelligence, but the function of ethics is overemphasised compared to other 
regional priorities.
- The Global Partnership on Artificial Intelligence Executive Council could increase the likelihood of Southeast Asian participation by amending its membership process to focus on artificial intelligence rather than national political systems.
- The Global Partnership on Artificial Intelligence could increase the likelihood of Southeast Asian participation by amending its Terms of Reference to allow for other intergovernmental organisations to join, in addition to the European Union.
 - The Global Partnership on Artificial Intelligence could improve the regional balance of its membership by recruiting the five G20 members who are not currently participants.

## Citation

If this work is useful to you, please cite the following paper: Keith, A.J. (2024) Governance of artificial intelligence in Southeast Asia. _Global Policy_, 00, 1–18. Available from: [https://doi.org/10.1111/1758-5899.13458](https://onlinelibrary.wiley.com/doi/10.1111/1758-5899.13458).

```
@article{https://doi.org/10.1111/1758-5899.13458,
author = {Keith, Andrew J.},
title = {Governance of artificial intelligence in Southeast Asia},
journal = {Global Policy},
volume = {n/a},
number = {n/a},
pages = {},
doi = {https://doi.org/10.1111/1758-5899.13458},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/1758-5899.13458},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/1758-5899.13458},
}
```
