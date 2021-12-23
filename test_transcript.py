import glob
import re
import numpy as np
import pandas as pd
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

### speaker functions ###

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

t_file = get_designated_text(raw_file)

# sentences 
s_trans = []
for sen in t_file:
    sen_i = re.sub(r'[^a-zA-Z]', ' ', sen)
    s_trans.append(re.sub(r' Speaker   ', '', sen_i))

# convert file to one full convo string
convo = []
for sen in t_file:
    convo.append(re.sub(r'\[Speaker [0-9]\]', '', sen))
convo = ''.join(convo)

def sentence_sim(transcript_sentences):
    
    print('Computing sentence scores')
    
    n = len(transcript_sentences)
    sent_scores = np.zeros((n, n))
    s_iter = 1
    
    for i in range(len(s_trans)):
        for j in range(len(s_trans)):
            if i != j:

                sen_i = [w.lower() for w in s_trans[i].split()]
                sen_j = [w.lower() for w in s_trans[j].split()]
                w_diff = list(set(sen_i + sen_j))
                w_dist_i = np.repeat(0, len(w_diff))
                w_dist_j = np.repeat(0, len(w_diff))

                for w in sen_i:
                    if not w in stop_words:
                        w_dist_i[w_diff.index(w)] += 1

                for w in sen_j:
                    if not w in stop_words:
                        w_dist_j[w_diff.index(w)] += 1

                sent_scores[i, j] = 1 - cosine_distance(w_dist_i, w_dist_j)
                print('   ' + str(s_iter) + ' of ' + str(n**2 - n) + ' complete')
                s_iter += 1
                
    return sent_scores

def get_summary(transcript_sentences, summary_n):
    
    similarity = sentence_sim(transcript_sentences)
    sent_scores = nx.pagerank(nx.from_numpy_array(similarity))

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
    return text_summary
    
get_summary(t_trans, 3)





# other insights - sentiment over time, speaker time, etc
                         


# make dict with sent dfs
speaker_dict = {}
for s in get_speakers(t_file)[0]:
    print('set ' + s + ' dict')
    speaker = re.sub(r' ', '_', s)
    speaker_val = int(s.split()[1])
    speaker_text = get_speaker_text(speaker_val, t_file)
    speaker_dict[speaker] = get_sen_df(speaker_text, speaker)

speaker_dict['Speaker_1']

# plot sentiment over speaker time


# speaker rate

s1 = get_speaker_text(1, t_file)
s2 = get_speaker_text(2, t_file)

s1_rate = len(''.join(s1).split())/(len(''.join(s1).split()) + len(''.join(s2).split()))

