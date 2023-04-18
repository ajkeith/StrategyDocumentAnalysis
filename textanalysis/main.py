import os
from textanalysis import analysis
import pandas as pd

# full loop
dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
texts, filenames, languages = analysis.load_texts(dir_path)
sclass, zclass = analysis.build_nlp_pipelines()
candidate_labels = ["development", "control", "promotion"]
figs = []
for text, filename, language in zip(texts, filenames, languages):
    if language == 'en':
        df = analysis.analyze_text(text, filename.replace('.txt', '.pkl'), sclass, zclass, candidate_labels)
        country = filename.split('_')[0]
        doctype = filename.split('_')[1].replace('.txt', '')
        fig = analysis.plot_nlp(df, country, doctype)
        figs.append(fig)
figs[0].show()

# exploration
from transformers import pipeline # nlp models
sclass = pipeline(task="sentiment-analysis")
zclass = pipeline(model="facebook/bart-large-mnli")
df_single = analysis.analyze_text(text, 'aus_gov_single_label.pkl', 
                           sclass, zclass, candidate_labels, 
                           step=500, multi_label=False)
fig_single = analysis.plot_nlp(df_single)
df_multi = analysis.analyze_text(text, 'aus_gov_multi_label.pkl', 
                           sclass, zclass, candidate_labels, 
                           step=500, multi_label=True)
fig_multi = analysis.plot_nlp(df_multi)
fig_single.show()
fig_multi.show()
df_multi.groupby('Label').mean().sort_values('Score', ascending=False)

for f in figs:
    f.show()