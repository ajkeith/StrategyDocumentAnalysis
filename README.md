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

`distilbert-base-uncased-finetuned-sst-2-english` has [strong evaluation results](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) in terms of accuracy and precision:

<img src="https://user-images.githubusercontent.com/26749415/230005604-98fd3980-7d26-48b9-92f8-82197664a339.png" width="500">

However, it is also subject to [risks, limitations, and biases](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english#risks-limitations-and-biases). 

## Initial Results

This projec tis still in progress, but intial results of exploratory analysis suggest that the approach is feasible. The following figure shows the sentiment and topic classficiation through Singapore's National AI Strategy.

![newplot (1)](https://user-images.githubusercontent.com/26749415/230006657-b511a380-7872-41ac-bd6b-daf230fb9790.png)
