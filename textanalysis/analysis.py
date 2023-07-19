import os
from transformers import pipeline # nlp models
import numpy as np # basic math
import pandas as pd # dataframes
import plotly.express as px # plotting
from pdfminer.high_level import extract_text # pdf read
from cleantext import clean # text cleaning

def extract_pdfs(dir_path):
    """
    Extract text from pdf
    """
    print("extracting text from pdfs...\n")
    directory = os.fsencode(os.path.join(os.getcwd(), 'data', 'policies'))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"): 
            print("Converting " + filename + " ...")
            fileloc = os.path.join(os.getcwd(), 'data', 'policies', filename)
            outname = os.path.join(os.getcwd(), 'data', 'texts', filename.replace(".pdf",".txt"))
            with open(outname, "w", encoding='utf-8') as text_file:
                text = extract_text(fileloc)
                text_file.write(text)
            continue
        else:
            continue
    print("text extract complete\n")
    return text

def load_texts(dir_path, clean_flag=True):
    texts = []
    filenames = []
    languages = []
    for file in os.listdir(dir_path):
        filename = os.fsdecode(file)
        if filename.endswith("governance.txt"):
            languages.append('id') if filename.startswith('indonesia') else languages.append('en')
            with open(os.path.join(os.getcwd(), 'data', 'texts', filename), 'r', encoding='utf-8') as f:
                raw_text = f.read().replace('\n', '')
                clean_text = clean(raw_text,
                                no_line_breaks=True,
                                no_urls=True,
                                no_digits=True,
                                no_emails=True,
                                no_phone_numbers=True,
                                no_numbers=True)
                texts.append(clean_text) if clean_flag else texts.append(raw_text)
                filenames.append(filename)
    return texts, filenames, languages

def build_nlp_pipelines():
    """
    Build sentiment and zero shot topic classification pipelines.
    """
    # sentiment pipeline model="distilbert-base-uncased-finetuned-sst-2-english"
    # note: direclty downloading the same classifier weights is timing out for some reason
    print("building sentiment pipeline...\n")
    sclass = pipeline(task="sentiment-analysis")
    print("building zero shot topic classification pipeline...\n")
    zclass = pipeline(model="facebook/bart-large-mnli")
    print("pipelines complete\n")
    return sclass, zclass

def analyze_text(text, name, sclass, zclass, candidate_labels, step=500, multi_label=True):
    """
    Analyze text in chunks and return a dataframe with topic and sentiment
    """
    nlabel = len(candidate_labels)
    topics = []
    maxidx = len(text)
    idx = 0
    print("analyzing text...")
    while (idx+step) < maxidx:
        sentiment = sclass(text[idx:(idx+step)])
        topic = zclass(text[idx:(idx+step)],
            candidate_labels,
            multi_label=multi_label)
        idx += step
        srow = [(idx, 'sentiment', sentiment[0]['score'])]
        topics.extend(srow)
        zrow = list(zip([idx]*nlabel, topic['labels'], topic['scores']))
        topics.extend(zrow)
        # print progress bar
        print(f"Progress: {idx/maxidx:.2%}", end="\r")
    df = pd.DataFrame(topics, columns=['Index', 'Label', 'Score'])
    out_path = os.getcwd() + '\\data\\output\\' + name
    df.to_pickle(out_path)
    print("\ntext analysis complete\n")
    return df

def plot_nlp(df, country='', doctype='governance'):
    """
    Plot sentiment and topic classification
    """
    print("plotting results...\n")
    title = 'Topic and Sentiment: ' + country.capitalize()
    match doctype:
        case 'governance':
            title += ' AI Governance'
        case 'strategy':
            title += ' Defense Strategy'
        case 'ethics':
            title += ' AI Ethics'
        case _:
            title += ''
    fig = px.scatter(df, x='Index', y='Score', color='Label', 
                    labels=dict(Index='Text Position Index', Label='Topic'),
                    title='Topic and Sentiment: Indonesia National AI Strategy',
                    trendline='lowess', trendline_options=dict(frac=0.2),
                    template="simple_white")
    fig.data = [t for t in fig.data if t.mode == 'lines']
    fig.update_traces(showlegend=True)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    ))
    return fig

def analyze_corpus(text_data, candidate_labels):
    """
    Analyze a corpus of pdfs and return a dataframe with topic and sentiment
    """
    # TODO: finish loop
    sclass, zclass = build_nlp_pipelines()
    df = analyze_text(text_data, 'text_data.pkl', sclass, zclass, candidate_labels)
    fig = plot_nlp(df)
    fig.show()
    return df, fig