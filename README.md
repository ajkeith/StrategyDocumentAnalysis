# AI and Defense Strategy: Text Analysis

[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/621097589)

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

Exploratory analysis suggest that the approach is feasible. The following figure shows the sentiment and topic classficiation through Singapore's National AI Strategy.

![text classificaiton figure](https://user-images.githubusercontent.com/26749415/230006657-b511a380-7872-41ac-bd6b-daf230fb9790.png)
