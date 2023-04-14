from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# https://stackoverflow.com/questions/65113929/how-sklearn-latent-dirichlet-allocation-really-works
# TODO: LDA

from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
import os

n_samples = 2000
n_features = 1000
n_components = 5
n_top_words = 10

# load text files
directory = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
texts = []
filenames = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # if (filename.endswith("defense.txt") or filename.endswith("governance.txt")) and not filename.startswith("indonesia"): 
    if filename.endswith("governance.txt") and not filename.startswith("indonesia"): 
        with open(os.path.join(os.getcwd(), 'data', 'texts', filename), 'r', encoding='utf-8') as f:
            texts.append(f.read().replace('\n', ''))
            filenames.append(filename)
len(texts)
filenames

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    

#create a count vectorizer using the sklearn CountVectorizer which has some useful features

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
                                            'vietnam'])
tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words=list(stop_words_custom))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

X = texts
vectorizedX = tf_vectorizer.fit_transform(X)
lda.fit(vectorizedX)
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, n_top_words)
[lda.transform(tf_vectorizer.transform([t])).max() for t in texts]
lda.transform(tf_vectorizer.transform(["This is a strategy documenta about AI manufacturing"])).max()