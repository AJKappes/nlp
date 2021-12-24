import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
si = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

########

sub_path = glob.glob('/home/alex/pyenv/nlp/*.txt')
t_path = [t for t in sub_path if re.search(r'transc', t)][0]
raw_file = open(t_path).readlines()

########

def get_designated_text(file):
    
    # filter only speaker lines
    speaker_lines = [s for s in file if re.search(r'^\[Speaker [0-9]\]', s)]
    # filter out inaudible sections
    clean_inaud = [re.sub(r'\[\bInaudible\b \d+.\d+.\d+\]', '', s) for s in speaker_lines]
    # clean line breaks
    designated_text = [re.search(r'^.*', s)[0] for s in clean_inaud]
    
    return designated_text

# speaker functions

def get_speakers(file):
    
    speaker_num = []
    for i in range(len(t_file)):
        speaker_num.append(re.search(r'Speaker [0-9]', file[i])[0])
    
    speaker_list = list(set(speaker_num))
    n = len(set(speaker_num))
    
    return [speaker_list, n]

def get_speaker_text(num, file):
    
    speaker = [s for s in file if re.search('Speaker ' + str(num), s)]
    
    speaker_clean = []
    for i in range(len(speaker)):
        speaker_clean.append(re.sub(r'\[Speaker ' + str(num) + '\]', '', speaker[i]))
    
    return speaker_clean

def get_speaker_rates(file):
    
    n_speakers = get_speakers(file)[1]
    
    speaker_text = []
    for s in range(1, n_speakers + 1):
        speaker_text.append(get_speaker_text(s, t_file))
    
    s_word_count = [len(''.join(s).split()) for s in speaker_text]
    s_rates = [s_words/sum(s_word_count) for s_words in s_word_count]
    
    return s_rates

def get_sen_df(s_text, speaker):
    
    s_df = pd.DataFrame(s_text, columns = [speaker])
    s_sen = [*s_df[speaker].apply(si.polarity_scores)]
    
    s_senval = []
    for i in range(len(s_sen)):
        print('collecting sen val ' + str(i))
        s_senval.append(s_sen[i]['compound'])
    
    s_df['sent'] = s_senval
    return s_df

# extractive summary 

def get_summary(transcript_sentences, summary_n):
    
    # compute sentence similarity matrix
    # textrank through pagerank (cosine distance)
    
    print('Computing sentence similarity')
    n = len(transcript_sentences)
    sent_sim = np.zeros((n, n))
    s_iter = 1
    
    for i in range(len(s_trans)):
        for j in range(len(s_trans)):
            if i != j:
                
                # i, j sentence comparison in nxn similarity matrix
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
        summary.append(transcript_sentences[key_val])
    
    text_summary = ' '.join(summary)
    print(str(summary_n) + ' line summary extracted')
    
    return text_summary    

########

t_file = get_designated_text(raw_file)

# convert file to one full convo string
convo = []
for sen in t_file:
    convo.append(re.sub(r'\[Speaker [0-9]\]', '', sen))
convo = ''.join(convo)

# sentences 
s_trans = []
for sen in t_file:
    sen_i = re.sub(r'[^a-zA-Z]', ' ', sen)
    s_trans.append(re.sub(r' Speaker   ', '', sen_i))

get_summary(s_trans, 3)
get_speaker_rates(t_file)

########
                         
# make dict with sent dfs
speaker_dict = {}
for s in get_speakers(t_file)[0]:
    print('set ' + s + ' dict')
    speaker = re.sub(r' ', '_', s)
    speaker_val = int(s.split()[1])
    speaker_text = get_speaker_text(speaker_val, t_file)
    speaker_dict[speaker] = get_sen_df(speaker_text, speaker)

# plot sentiment over speaker time

x = speaker_dict['Speaker_1'].index.tolist()
sent_vals = [speaker_dict[s]['sent'].values for s in speaker_dict.keys()]

fig, ax = plt.subplots()
ax.plot(x, sent_vals[0], label = 'Speaker 1')
ax.plot(x, sent_vals[1], label = 'Speaker 2')
ax.set_ylabel('Sentiment')
ax.set_ylim([-1, 1])
ax.set_xticks([])
ax.set_title('Individual Sentiment Across Meeting')
plt.legend()

# speaker rate vis

speakers = get_speakers(t_file)[0]
rates = get_speaker_rates(t_file)
colors = ['#ff9999','#66b3ff']

fig, ax = plt.subplots()
ax.pie(rates, colors = colors, labels = speakers,
       autopct = '%1.1f%%', startangle = 90, explode = (.01, .01))

center_cir = plt.Circle((0,0), .85, fc = 'white')
fig = plt.gcf()
fig.gca().add_artist(center_cir)

ax.axis('equal')
plt.tight_layout()
plt.show()