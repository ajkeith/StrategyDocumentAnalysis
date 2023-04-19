import os
from textanalysis import analysis
import pandas as pd
from transformers import pipeline # nlp models
from pdfminer.high_level import extract_text # pdf read
import plotly.express as px

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
df_id.groupby('Label').mean().drop('Index', axis=1)
fig_id = analysis.plot_nlp(df_id, 'indonesia (English Topics)', 'governance')
fig_id.show()
df_id_lbl = analysis.analyze_text(texts[2], 'indonesia_governance_id.pkl', 
                           sclass, iclass, candidate_labels_id, 
                           step=500, multi_label=True)
df_id_lbl.groupby('Label').mean().drop('Index', axis=1)
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
df_th_en.groupby('Label').mean().drop('Index', axis=1)
fig_th_en = analysis.plot_nlp(df_th_en, 'thailand (English Topics)', 'governance')
fig_th_en.show()
df_th_th = analysis.analyze_text(text_th, 'thailand_governance_th_th.pkl', 
                           sclass, iclass, candidate_labels_th, 
                           step=500, multi_label=True)
df_th_th.groupby('Label').mean().drop('Index', axis=1)
fig_th_th = analysis.plot_nlp(df_th_th, 'thailand (Thai Topics)', 'governance') 
fig_th_th.show()

# classify alternative australian governance text
filepath = os.fsencode(os.path.join(os.getcwd(), 'data', 'policies','australia_governance_alt.pdf'))
filename = 'australia_governance_alt.pdf'
outname = os.path.join(os.getcwd(), 'data', 'texts', filename.replace(".pdf",".txt"))
with open(outname, "w", encoding='utf-8') as text_file:
    text = extract_text(os.path.join(os.getcwd(), 'data', 'policies', filename))
    text_file.write(text)
dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts', 'australia_governance_alt.txt'))
with open(dir_path, 'r', encoding='utf-8') as f:
    text_au_alt = f.read().replace('\n', '')
candidate_labels = ["development", "control", "promotion"]
df_au_alt = analysis.analyze_text(text_au_alt, 'australia_governance_alt.pkl', 
                           sclass, zclass, candidate_labels, 
                           step=500, multi_label=True)
df_au_alt.groupby('Label').mean().drop('Index', axis=1)
fig_au_alt = analysis.plot_nlp(df_au_alt, 'australia (alternate)', 'governance')
fig_au_alt.show()

# summary table of all countries
frames = []
rawdfs = []
dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'output'))
for f in os.listdir(dir_path):
    filename = os.fsdecode(f)
    if filename.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'output', filename))
        df_summary = df.groupby('Label').mean().drop('Index', axis=1)
        country = filename.split('_')[0]
        df_summary['Country'] = [country] * 4
        rawdfs.append(df_summary)
df_summary = pd.concat(rawdfs)
label_tx = {'perkembangan': 'development', 'pengendalian': 'control', 'promosi': 'promotion',
            'การพัฒนา': 'development', 'การควบคุม': 'control', 'การส่งเสริม': 'promotion'}
df_full = df_summary.reset_index().replace(label_tx)
df_plot = df_full[~df_full.Label.str.contains('sentiment')]
fig = px.bar(df_plot, x='Country', y='Score', color='Label', barmode='group', orientation='h')
fig.show()
fig.write_html(os.path.join(os.getcwd(), 'data', 'output', 'figures', 'topic_summary.html'))
df_plot.to_csv(os.path.join(os.getcwd(), 'data', 'output', 'tables', 'topic_summary.csv'), index=False)
df_sea = df_plot[df_plot.Country.isin(['indonesia', 'malaysia', 'thailand', 'vietnam'])]
df_sea.groupby('Label').mean()
