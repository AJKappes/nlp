import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from nltk.cluster.util import cosine_distance
import networkx as nx
from collections import Counter
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
# si = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')
new_words = ['uh', 'um', 'yeah', 'yes', 'gon', 'na', 'ca', 'n\'t',
             'i', 'meeting', 'today', 'okay', 'uh,', 'hi', 'hello',
             'bye']
for w in new_words:
    stop_words.append(w)

# abstractive vs extractive
# abstractive limitation is that there exists no training summary
# also difficult is that this is a speech to text meeting transcript
#  not a written piece and progression may not be as clear/logical/informative

# TODO 
#    include sourced literature on extractive summary methodology
#        (textrank, others)
#    performance measures
#        (precision, recall, F-score as f(precision, recall))
#        but these require training summaries to compute


######## read in data ########


sub_path = glob.glob('/home/alex/pyenv/nlp/read/Recruiting/*.txt')
t_path = [t for t in sub_path if re.search(r'Product', t)][0]
raw_file = open(t_path, 'r').readlines()


######## helper functions ########


def process_file(file):
    # function processes file columns into distinct lists for analysis
    
    # args:
    #    meeting transcipt file
    
    names = []
    timestamp = []
    talk = []
    for i in file:
        if re.search(r'^.*[A-Za-z](?=:)', i):
            names.append(re.search(r'^.*[A-Za-z](?=:)', i)[0])

        if re.search(r'\d+:\d+', i):
            timestamp.append(re.search(r'\d+:\d+', i)[0])

        if re.search(r'^.*\d+:\d+\s+(.*)', i):
            talk.append(re.search(r'^.*\d+:\d+\s+(.*)', i)[1])
    
    talk_cleaned = [re.sub(r'\[\bcrosstalk\b \d+.\d+.\d+\]', '', t) for t in talk]
    
    return {'names': names, 'timestamp': timestamp, 'talk': talk_cleaned}

def get_summary_textrank(transcript, summary_n):
    # function builds extractive summary
    #    top n sentences based on textrank
    #    scores computed through sentence cosine distance similiraity matrix
    
    # args:
    #    transcript_sentences, list of meeting text
    #    summary_n, the number of sentences to build the summary
    
    s_trans = transcript
    
    # sentence similarity (cosine distance)
    print('Computing sentence similarity')
    n = len(s_trans)
    sent_sim = np.zeros((n, n))
    s_iter = 1
    
    for i in range(len(s_trans)):
        for j in range(len(s_trans)):
            if i != j:
                
                # i, j sentence comparison in nxn similarity matrix
                # computed using whole words NOT sentence characters
                sen_i = [w.lower() for w in s_trans[i].split()]
                sen_j = [w.lower() for w in s_trans[j].split()]
                w_diff = list(set(sen_i + sen_j))
                w_dist_i = np.repeat(0, len(w_diff))
                w_dist_j = np.repeat(0, len(w_diff))
                
                # index values for !stopwords
                for w in sen_i:
                    if not w in stop_words:
                        w_dist_i[w_diff.index(w)] = 1

                for w in sen_j:
                    if not w in stop_words:
                        w_dist_j[w_diff.index(w)] = 1
                
                # similarity values
                sent_sim[i, j] = 1 - cosine_distance(w_dist_i, w_dist_j)
                print('   ' + str(s_iter) + ' of ' + str(n**2 - n) + ' complete')
                s_iter += 1
    
    print('\nSentence similarity matrix complete\n')
    
    # extractive summary
    sent_scores = nx.pagerank(nx.from_numpy_array(sent_sim))
    score_val = list(sent_scores.values())
    score_key = list(sent_scores.keys())
    s_rank = sorted(score_val, reverse=True)[:summary_n]
    
    summary = []
    for i in range(summary_n):
        # get index of scored sentence
        val_idx = score_val.index(s_rank[i])
        # get dict key for sentence - idx for all scored sentence
        key_val = score_key[val_idx]
        # append summary sentences
        summary.append(s_trans[key_val])
    
    text_summary = ' '.join(summary)
    print(str(summary_n) + ' line summary extracted')
    
    return text_summary

def get_summary_kmeans(transcript, summary_n):
    # function builds summary through K means clustering
    #  built through word2vec embeddings mapped to sentence vectors
    #  sentence vectors then used to build clusters and extract summary
    
    # args:
    #    transcript_sentences, list of speaker text
    #    summary_n, the number of sentences to build the summary
    
    meeting_text = ' '.join(transcript)
    sentences = [s.lower() for s in sent_tokenize(meeting_text)]
    pre_convo = []

    for sentence in sentences:
        pre = re.sub(r'[^a-z]', ' ', sentence)
        words = pre.split()
        keep_words = [w for w in words if w not in stop_words]
        new_sen = ' '.join(keep_words)
        pre_convo.append(new_sen)

    content_idx = []
    for i in range(len(pre_convo)):
        if pre_convo[i] != '':
            content_idx.append(i)

    sentence_extract = [sentences[i] for i in content_idx]
    clean_convo = [sen for sen in pre_convo if re.search(r'[a-z]', sen)]
    convo_embed = [i.split() for i in clean_convo]
    embed_dim = max([len(i) for i in convo_embed]) + 1
    embed_model = Word2Vec(convo_embed, min_count = 1, vector_size = embed_dim)

    # averaged word2vec for sentence embedding
    sentence_embed = []
    for sentence in convo_embed:

        summed = 0
        for w in sentence:
            summed += embed_model.wv[w]

        s_embed = summed/len(sentence)
        sentence_embed.append(s_embed)

    kmeans = KMeans(summary_n, random_state = 1)
    kmeans_fit = kmeans.fit_predict(sentence_embed)

    sentence_idx = []
    for c in range(summary_n):

        idx = np.where(kmeans_fit == c)[0].tolist()
        distances = []
        for i in idx:
            d = distance.euclidean(kmeans.cluster_centers_[c], sentence_embed[i])
            distances.append(d)

        #dsort = sorted(distances)[:2]
        #didx = [distances.index(d) for d in dsort]
        #sentence_idx.append([d for d in didx])
        didx = distances.index(min(distances))
        sentence_idx.append(didx)
    
    sentence_summary = [sentence_extract[i] for i in sentence_idx]
    summary = ' '.join(sentence_summary)

    return summary

def get_word_freqs(talk_text):
    # function computes n-gram word frequencies
    
    # args:
    #    talk_text, list of meeting text
    
    meeting_text = ' '.join(talk_text)
    tokens = [w.lower() for w in word_tokenize(meeting_text)]
    words = [w for w in tokens if w not in stop_words]
    cleaned = [w for w in words if re.search(r'^[a-z]', w)]
    
    grams = ['1', '2', '3', '4', '5', '6']
    freq_dict = {}
    for key in grams:
        key_df = pd.DataFrame(Counter(ngrams(cleaned, int(key))).items(),
                              columns = ['words', 'freq'])
        freq_dict[key] = key_df.sort_values(by = 'freq', ascending = False)
    
    return freq_dict

# speaker rate summary

def get_speaker_info():
    # function provides speaker and speaker rate information
    #  also provided is a dataframe with all above information
    
    df = pd.DataFrame({'speaker': names, 'text': talk})
    speakers = df['speaker'].unique().tolist()
    N_words = len(df['text'].str.cat().split())
    
    rates = []
    for speaker in speakers:
        word_count = len(df.loc[df['speaker'] == speaker,
                                'text'].str.cat().split())
        rates.append(word_count/N_words)
    
    return {'speakers': speakers, 'rates': rates, 'df': df}


######## data vis ########


processed_file = process_file(raw_file)
talk = processed_file['talk']
names = processed_file['names']

# n-gram vis
word_freqs = get_word_freqs(talk)
grams = ['2', '4', '5']
ydata = []
xdata = []
for g in grams:
    ydata.append([*word_freqs[g]['words'].head()])
    xdata.append(word_freqs[g]['freq'].head().values)

fig, axs = plt.subplots(3)
fig.suptitle('Meeting n-gram Frequencies')
axs[0].barh([str(i) for i in ydata[0]],
            xdata[0],
            alpha = 0.7)
axs[1].barh([str(i) for i in ydata[1]],
            xdata[1],
            color = 'red',
            alpha = 0.7)
axs[2].barh([str(i) for i in ydata[2]],
            xdata[2],
            color = 'green',
            alpha = 0.7)
axs[0].invert_yaxis()
axs[1].invert_yaxis()
axs[2].invert_yaxis()
plt.xlabel('Frequency')
plt.subplots_adjust(hspace = .5)
plt.show()

# speaker rate vis
speakers = get_speaker_info()['speakers']
rates = get_speaker_info()['rates']
colors = ['#38387b', '#58b6ba', '#ccf7c7',
          '#edd752', '#ec912e', '#ec322e']

fig, ax = plt.subplots()
ax.pie(rates, colors = colors, labels = speakers,
       autopct = '%1.1f%%', startangle = 90)

center_cir = plt.Circle((0,0), .85, fc = 'white')
fig = plt.gcf()
fig.gca().add_artist(center_cir)

ax.axis('equal')
plt.tight_layout()
plt.show()


######## meeting summary ########


# summary based on ALL meeting speaers
speakers_all_summary = get_summary_textrank(talk, 2)

# summary based on MOST involved speakers
df = get_speaker_info()['df']
sort_rates = sorted(rates, reverse = True)[:2]
summary_speakers = []
for r in sort_rates:
    summary_speakers.append(speakers[rates.index(r)])

sub_talk = df.loc[df['speaker'].isin(summary_speakers), 'text'].tolist()
speakers_most_summary = get_summary_textrank(sub_talk, 2)

# summary based on K means clustering
kmeans_summary = get_summary_kmeans(talk, 4)

print('All speaker summary:\n\n  ', speakers_all_summary, '\n\n',
      'Most speaker summary:\n\n  ', speakers_most_summary, '\n\n',
      'K means summary:\n\n  ', kmeans_summary)


