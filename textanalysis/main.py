import os
from textanalysis import analysis
import pandas as pd
from cleantext import clean # text cleaning
from transformers import pipeline # nlp models

# classify english texts
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

# classify indonesia governance text
sclass = pipeline(task="sentiment-analysis")
iclass = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
candidate_labels_id = ["perkembangan", "pengendalian", "promosi"]
df_id = analysis.analyze_text(texts[2], 'indonesia_governance.pkl', 
                           sclass, iclass, candidate_labels, 
                           step=500, multi_label=True)
fig_id = analysis.plot_nlp(df_id, 'indonesia (English Topics)', 'governance')
fig_id.show()
df_id_lbl = analysis.analyze_text(texts[2], 'indonesia_governance_id.pkl', 
                           sclass, iclass, candidate_labels_id, 
                           step=500, multi_label=True)
fig_id_lbl = analysis.plot_nlp(df_id_lbl, 'indonesia (Bahasa Topics)', 'governance')
fig_id_lbl.show()

# classify thai governance text
dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
with open(os.path.join(os.getcwd(), 'data', 'texts', 'thailand_governance_thai.txt'), 'r', encoding='utf-8') as f:
    text_th = f.read().replace('\n', '')
candidate_labels = ["development", "control", "promotion"]
candidate_labels_th = ["การพัฒนา", "การควบคุม", "การส่งเสริม"]
df_th_en = analysis.analyze_text(text_th, 'thailand_governance_th_en.pkl', 
                           sclass, iclass, candidate_labels, 
                           step=500, multi_label=True)
fig_th_en = analysis.plot_nlp(df_th_en, 'thailand (English Topics)', 'governance')
fig_th_en.show()

df_th_th = analysis.analyze_text(text_th, 'thailand_governance_th_th.pkl', 
                           sclass, iclass, candidate_labels_th, 
                           step=500, multi_label=True)
fig_th_th = analysis.plot_nlp(df_th_th, 'thailand (Thai Topics)', 'governance') 
fig_th_th.show()

# table summary
df_id.groupby('Label').mean().sort_values('Score', ascending=False)
for f in figs:
    f.show()