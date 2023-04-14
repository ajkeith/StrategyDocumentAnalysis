import os
from transformers import pipeline # nlp models
import numpy as np # basic math
import pandas as pd # dataframes
import plotly.express as px # plotting
from pdfminer.high_level import extract_text # pdf read

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

def build_nlp_pipelines():
    """
    Build sentiment and zero shot topic classification pipelines
    """
    # sentiment pipeline model="distilbert-base-uncased-finetuned-sst-2-english"
    # note: direclty downloading the same classifier weights is timing out for some reason
    print("building sentiment pipeline...\n")
    sclass = pipeline(task="sentiment-analysis")
    print("building lda pipeline...\n")
    # TODO: topic pipeline with lda
    print("building zero shot topic classification pipeline...\n")
    zclass = pipeline(model="facebook/bart-large-mnli")
    print("pipelines complete\n")
    return sclass, zclass

def analyze_text(text, name, sclass, zclass, candidate_labels, step=500):
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
            multi_label=True)
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

def plot_nlp(df):
    """
    Plot sentiment and topic classification
    """
    print("plotting results...\n")
    fig = px.scatter(df, x='Index', y='Score', color='Label', 
                    labels=dict(Index='Text Position Index', Label='Topic'),
                    title='Natural Language Processing: Singapore Naitonal AI Strategy',
                    trendline='lowess', trendline_options=dict(frac=0.2))
    fig.data = [t for t in fig.data if t.mode == 'lines']
    fig.update_traces(showlegend=True)
    return fig

def analyze_corpus(text_data):
    """
    Analyze a corpus of pdfs and return a dataframe with topic and sentiment
    """
    # TODO: finish loop
    sclass, zclass = build_nlp_pipelines()
    candidate_labels = ["artificial intelligence", 
                        "governance", "ethics", 
                        "defence", "security"]
    # df = pd.read_pickle("text_data.pkl") # replace with actual pkl files
    df = analyze_text(text_data, 'text_data.pkl', sclass, zclass, candidate_labels)
    fig = plot_nlp(df)
    fig.show()
    return df, fig