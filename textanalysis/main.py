import os
from textanalysis import analysis

filepath = os.path.join(os.getcwd(), 'data', 'texts', 'australia_governance.txt')
with open(filepath, encoding='utf-8') as f:
    text = f.read().replace('\n', '')
candidate_labels = ["development", "control", "promotion"]
df, fig = analysis.analyze_corpus(text, candidate_labels)

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