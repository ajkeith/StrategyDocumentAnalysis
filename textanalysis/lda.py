from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
from cleantext import clean
import os

# set parameters
n_samples = 2000
n_features = 1000
n_components = 3
n_top_words = 10

# load text files
directory = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
texts = []
filenames = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # if (filename.endswith("defense.txt") or filename.endswith("governance.txt")) and not filename.startswith("indonesia"): 
    if filename.endswith("governance.txt") and not (filename.startswith("gpai") or filename.startswith("indonesia")): 
        with open(os.path.join(os.getcwd(), 'data', 'texts', filename), 'r', encoding='utf-8') as f:
            clean_text = clean(f.read().replace('\n', ''),
                               no_line_breaks=True,
                               no_urls=True,
                               no_digits=True,
                               no_emails=True,
                               no_phone_numbers=True,
                               no_numbers=True)
            texts.append(clean_text)
            filenames.append(filename)
len(texts)
filenames

def print_top_words(model, feature_names, n_top_words):
    """
    Print the top words for each topic.
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# build LDA model
stop_words_custom = text.ENGLISH_STOP_WORDS.union(['australia', 
                                            'brunei', 
                                            'cambodia', 
                                            'indonesia', 
                                            'laos', 
                                            'malaysia', 
                                            'myanmar', 
                                            'philippines', 
                                            'singapore', 
                                            'thailand', 
                                            'vietnam',
                                            'viet',
                                            'nam',
                                            '<number>',
                                            '<email>',
                                            '<url>',
                                            '<phone>',
                                            '00'])
tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words=list(stop_words_custom))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

# classify topics
X = texts
vectorizedX = tf_vectorizer.fit_transform(X)
lda.fit(vectorizedX)
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, n_top_words)
topics = [lda.transform(tf_vectorizer.transform([t])).argmax() for t in texts]
probs = [lda.transform(tf_vectorizer.transform([t])).max() for t in texts]
topic_summary = list(zip(filenames, topics, probs))
print(*topic_summary, sep='\n')