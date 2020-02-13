## Author:  Joe Price
## Desc:    Code used for IST736 homework #2 & 3
import datetime
import pandas as pd
import numpy as np
import re
import string
import os
import nltk
#nltk.download("stopwords")
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
## Add some custom stop words:
STOPWORDS2 = ["said","says","machine","learning","artificial","intelligence","ai","via","will"] + list(STOPWORDS)

## =========================================
## Corpus Creation
## This will (should) only run once as long as the corpus folder exists
## Make sure that the raw corpus file is in the same directory as this script
corpus_dir, corpus_file = "ai_corpus", "corpus_raw.txt"
## Check for corpus folder:
if not os.path.isdir(corpus_dir) and os.path.exists(corpus_file):
    ## Create the corpus directory
    os.mkdir(corpus_dir)
    ## Open the raw condensed corpus data from sketchengine
    corpus_raw = open(corpus_file, "r", encoding="utf-8")
    outlet_ctr = {}
    corpus_out = ""
    for line in corpus_raw:
        if line.startswith("<article"):
            ## New article
            ## Get the media outlet
            outlet = re.findall(r'\w+(?=.com)', line)[0]
            ## Add to the outlet counter
            if outlet not in outlet_ctr:
                outlet_ctr[outlet] = 1
            else:
                outlet_ctr[outlet] += 1
            ## Set the new file name
            corpus_out = outlet + str(outlet_ctr[outlet]) + ".txt"
            ## Go to the next line to start writing the new file
            continue
        else:
            ## Continue writing article
            with open(corpus_dir + "\\" + corpus_out, "a", encoding="utf-8") as f:
                ## Write next line of article to file
                ## - Unless the line contains count stats inserted by sketchengine
                ## - Remove html tags and encoded characters (&nbsp; etc)
                ## - Remove end article tag: </article>
                ## - Remove ad lines; marked with 'AD' or 'Advertisement' by sketchengine
                f.write(re.sub(r'^<p> [-+]\d*.* <\/p>\n|^<p> (?:Advertisement|AD) <\/p>\n|^<p> | <\/p>$|<\/article>\n|&.*;', '',line))

## =========================================
## Corpus (Articles) Vectorization
corpus_rdr = nltk.corpus.PlaintextCorpusReader(corpus_dir, ".*\.txt")
## corpus_rdr.raw()  # <-- All text from all files in one object

## Using list comprehension, concatenate file name to path for full path strings
corpus_full_paths = [str(corpus_rdr.root) + "\\" + str(f) for f in corpus_rdr.fileids()]

## ------------------------------------------
## Term Frequency Vectorization

## Vectorize the words\tokens in the files
## - Resulting vector will contain unigrams only
corpus_vectorizer = CountVectorizer(input="filename")
corpus_vectors = corpus_vectorizer.fit_transform(corpus_full_paths)

## Store in a DataFrame
corpus_DF = pd.DataFrame(corpus_vectors.toarray(), columns=corpus_vectorizer.get_feature_names())
## Rename the records as file name
## - Using dictionary comprehension for mapping
corpus_DF.rename({k:list(corpus_rdr.fileids())[k].replace(".txt","") for k in range(0, len(list(corpus_rdr.fileids())))},
                 axis="index",
                 inplace=True)

## ------------------------------------------
## TFIDF Vectorization (Normalized)

## Vectorize the words\tokens in the files
## - Resulting vector will contain unigrams only
corpus_tfidf_vectorizer = TfidfVectorizer(input="filename")
corpus_tfidf_vectors = corpus_tfidf_vectorizer.fit_transform(corpus_full_paths)

## Store in a DataFrame
corpus_tfidf_DF = pd.DataFrame(corpus_tfidf_vectors.toarray(), columns=corpus_tfidf_vectorizer.get_feature_names())
## Rename the records as file name
## - Using dictionary comprehension for mapping
corpus_tfidf_DF.rename({k:list(corpus_rdr.fileids())[k].replace(".txt","") for k in range(0, len(list(corpus_rdr.fileids())))},
                 axis="index",
                 inplace=True)

## ------------------------------------------------
## Clean up the data
## Remove the following tokens from the list:
## - If token contains numeric characters
## - If in the WordCloud STOPWORDS library
## - If length is less than 3
for token in corpus_vectorizer.get_feature_names():
    if bool(re.search(r'\d', token)) or token in STOPWORDS2 or len(token) < 3:
        corpus_DF.drop(token, axis=1, inplace=True)
        corpus_tfidf_DF.drop(token, axis=1, inplace=True)

## Bar chart - articles by outlet
plt.figure(figsize=(10,5))
sns.countplot(x="Outlet", data=pd.DataFrame([re.sub(r'\d|\.txt', '',str(f)) for f in corpus_rdr.fileids()], columns=["Outlet"]))
plt.title("Articles per Outlet")
plt.savefig(fname="bar_outlets.png", bbox_inches="tight")

## ------------------------------------------------
## Word counts & Wordcloud visualizations
## Wordcloud of full corpus vocabulary (top 100 words)
corpus_freq = corpus_DF.sum(axis=0)
corpus_all_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(corpus_freq)
plt.figure()
plt.imshow(corpus_all_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Corpus Wordcloud")
plt.savefig(fname="wc_corpus.png", bbox_inches="tight")
print("===================\ncorpus term frequency")
print(corpus_freq.sort_values(ascending=False)[:10])

## Wordcloud of each outlet (top 100 words)
for outlet in np.unique([re.sub(r'\d|\.txt', '',str(f)) for f in corpus_rdr.fileids()]):
    outlet_freq = corpus_DF.filter(regex=outlet, axis=0).sum(axis=0)
    corpus_outlet_wc = WordCloud(max_font_size=100, max_words=100)\
       .generate_from_frequencies(outlet_freq)
    plt.figure()
    plt.imshow(corpus_outlet_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(outlet + " wordcloud")
    plt.savefig(fname="wc_" + outlet + ".png", bbox_inches="tight")
    print("===================\n" + outlet + " term frequency")
    print(outlet_freq.sort_values(ascending=False)[:10])

##------------------------------------------------
## K-Means clustering
## - Using the previously created TFIDF normalized vector

## Convert DF to matrix
corpus_matrix = corpus_tfidf_DF.values
print(corpus_tfidf_DF.values)

corpus_kmeans = KMeans(n_clusters=3)
corpus_kmeans.fit(corpus_matrix)
y_kmeans = corpus_kmeans.predict(corpus_matrix)
plt.scatter(corpus_matrix[:, 0], corpus_matrix[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = corpus_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("Corpus K-means, k=3")
plt.savefig(fname="kmeans_clusters.png", bbox_inches="tight")

## Show k-means calculation details
print("k-means with k = 5\n", corpus_kmeans.labels_)
corpus_kmeans_DF = pd.DataFrame([corpus_tfidf_DF.index, corpus_kmeans.labels_]).T
print("k-means results\n", corpus_kmeans_DF.to_string(index=False, header=False))

## =========================================
## CSV (Tweets) Vectorization

## Read the file into a DataFrame
tweets_DF = pd.read_csv("tweets-20200124-235150.txt")

## -----------------------------------------
## Tweet token count vectorizer
## Instantiate the vectorizer
tweets_vectorizer = CountVectorizer(binary=False, min_df=5)
## Run the vectorizer on the Tweet data
tweets_vector = tweets_vectorizer.fit_transform(tweets_DF["TweetText"].values)
## Replace the token ID with the token value
tweets_vector_DF = pd.DataFrame(tweets_vector.toarray(), columns=tweets_vectorizer.get_feature_names())

# ## ------------------------------------------------
# ## Clean up the data
# ## Remove the following tokens from the list:
# ## - If token contains numeric characters
# ## - If in the WordCloud STOPWORDS library
# ## - If length is less than 3
for token in tweets_vectorizer.get_feature_names():
    if bool(re.search(r'\d', token)) or token in STOPWORDS2 or len(token) < 3:
        tweets_vector_DF.drop(token, axis=1, inplace=True)

## ------------------------------------------------
## Word counts & Wordcloud visualizations
## Wordcloud of full corpus vocabulary (top 100 words)
tweets_freq = tweets_vector_DF.sum(axis=0)
tweets_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(tweets_freq)
plt.figure()
plt.imshow(tweets_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Tweets Wordcloud")
plt.savefig(fname="wc_tweets.png", bbox_inches="tight")
print("===================\nTweets term frequency")
print(tweets_freq.sort_values(ascending=False)[:10])

## ------------------------------------------------
## Add sentiment analysis to the tweet record
sid = SentimentIntensityAnalyzer()
sentimentScores = []
sentimentAry = []
for t in tweets_DF['TweetText']:
    pol_s = sid.polarity_scores(t)["compound"]
    sentimentScores.append(pol_s)
    if (pol_s > 0):
        sentimentAry.append("pos")
    elif (pol_s < 0):
        sentimentAry.append("neg")
    else:
        sentimentAry.append("neu")
tweets_DF["sentiment_score"] = sentimentScores
tweets_DF["sentiment"] = sentimentAry

## ------------------------------------------------
## Create bar plot of sentiment breakdown
sns.countplot(data=tweets_DF, x="sentiment")
plt.savefig(fname="bar_sentiments.png", bbox_inches="tight")

