import os
from textanalysis import analysis

path = os.path.join(os.getcwd(), 'data', 'policies', 'australia_defense.pdf')
temp = analysis.extract_pdfs(path)
df, fig = analysis.analyze_corpus(temp)
