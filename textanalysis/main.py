import os
from textanalysis import analysis
import pandas as pd
from cleantext import clean # text cleaning

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
# zclass = pipeline(model="facebook/bart-large-mnli")
iclass = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
candidate_labels_id = ["perkembangan", "pengendalian", "promosi"]
df_id = analysis.analyze_text(texts[2], 'id_test.pkl', 
                           sclass, iclass, candidate_labels, 
                           step=500, multi_label=True)
fig_id = analysis.plot_nlp(df_id, 'indonesia (English Topics)', 'governance')
fig_id.show()

df_id_lbl = analysis.analyze_text(texts[2], 'id_test_id_lbl.pkl', 
                           sclass, iclass, candidate_labels_id, 
                           step=500, multi_label=True)
fig_id_lbl = analysis.plot_nlp(df_id_lbl, 'indonesia (Bahasa Topics)', 'governance')
fig_id_lbl.show()

# classify thai governance text
dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
with open(os.path.join(os.getcwd(), 'data', 'texts', 'thai_governance_thai.txt'), 'r', encoding='utf-8') as f:
    raw_text = f.read().replace('\n', '')
    clean_text = clean(raw_text,
                                no_line_breaks=True,
                                no_urls=True,
                                no_digits=True,
                                no_emails=True,
                                no_phone_numbers=True,
                                no_numbers=True)
    text = clean_text
candidate_labels = ["development", "control", "promotion"]
candidate_labels_th = ["การพัฒนา", "การควบคุม", "การส่งเสริม"]
df_th_en = analysis.analyze_text(texts[2], 'id_test.pkl', 
                           sclass, iclass, candidate_labels, 
                           step=500, multi_label=True)
fig_id = analysis.plot_nlp(df_id, 'indonesia (English Topics)', 'governance')
fig_id.show()

df_id_lbl = analysis.analyze_text(texts[2], 'id_test_id_lbl.pkl', 
                           sclass, iclass, candidate_labels_id, 
                           step=500, multi_label=True)
fig_id_lbl = analysis.plot_nlp(df_id_lbl, 'indonesia (Bahasa Topics)', 'governance')
fig_id_lbl.show()



df_multi.groupby('Label').mean().sort_values('Score', ascending=False)
for f in figs:
    f.show()