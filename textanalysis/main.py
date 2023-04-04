import os
from textanalysis.pipeline import extract_text, analzye_corpus

path = os.getcwd() + '\\data\\policies\\australia_defense.pdf'
temp = extract_pdfs(path)
df, fig = analyze_corpus(temp)