# AI and Defense Strategy: Text Analysis

TODO: badges

This python project analyzes national AI and defense strategy documents using zero-shot text classification. 

## Getting Started

```python
python -m main.py
```

## Usage

```python
import os
from textanalysis import analysis

path = os.getcwd() + '\\data\\policies\\australia_defense.pdf'
temp = analysis.extract_pdfs(path)
df, fig = analysis.analyze_corpus(temp)
```

The result of `analyze_corpus` is a dataframe of classified text (by topic and sentiment) and an interactive plot of the topic and sentiment by text chunk. 

## Algorithm Details

This code uses the `distilbert-base-uncased-finetuned-sst-2-english` model from [Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). This is a fine-tuned model based on [DistilBERT](https://arxiv.org/abs/1910.01108) and used here for zero-shot text classification. 