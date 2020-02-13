## Author:  Joe Price
## Desc:    Code used for IST736 homework #2 & 3
import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
## Add some custom stop words:
STOPWORDS2 = ["said","says","will","food","restaurant"] + list(STOPWORDS)

## =============================================
## Part 1 - Data Import
## =============================================
src_file_name = "deception_data_converted_final.csv"
## ---------------------------------------------
## The source file is comma delimited *but* the text qualifiers are single quotes
## This causes all sorts of issues with delimiter interpretation which is a headache for parsing columns
## As a pre-import cleaning step, a modified version of  the file will be generated
## This will help with importing into a dataframe accurately later
if not os.path.exists(src_file_name.replace("final","edit")):
    src_file = open(src_file_name,"r")
    src_file_edit = open(src_file_name.replace("final","edit"), "w")
    for line in src_file.readlines():
        ## Remove the following from the line:
        ## - Trailing commas - Created from the excessive delimiters
        ## - Double quotes - Unnecessary for analysis and could cause problems anyway
        ## - Escape sequences - Apparently leftover from parsing methods
        line = re.sub(r',*$|"|\\\'', '', line)
        ## Replace the single-quote text identifiers with double-quotes
        line = re.sub(r',\'',',"', line)
        line = re.sub(r'\'$', '"', line)
        ## Write the new line
        src_file_edit.write(line)
    src_file.close()
    src_file_edit.close()
    ## Now, we have a more suitable file to import that won't cause fits later

## ---------------------------------------------
## Load the file to a DataFrame
reviews_df = pd.read_csv(src_file_name.replace("final","edit"))
## Have a look at the DataFrame details
# reviews_df.info()
## Here is sample of the first 5 records
# print(reviews_df.head())

## =============================================
## Part 2 - Data Cleaning & Preprocessing
## =============================================
## There are some reviews that are just a '?' character, let's remove those completely
bad_rows = reviews_df[reviews_df["review"] == "?"].index
reviews_df.drop(bad_rows, inplace=True)
reviews_df.reset_index(inplace=True)

## ---------------------------------------------
## Let's get ready to vectorize!!
## First, let's do a standard count vectorizer
## The count vectorizer will prune tokens if:
## - In the STOPWORDS2 collection
## - Numeric
reviews_vectorizer = CountVectorizer(min_df=3,
                                     stop_words=STOPWORDS2,
                                     token_pattern=r'\b[^\d\W]+\b'
                                     )
## Run the vectorizer on the review data
reviews_vector = reviews_vectorizer.fit_transform(reviews_df["review"].values)
## Replace the token ID in the column label with the token value (the actual word)
reviews_vector_df = pd.DataFrame(reviews_vector.toarray(), columns=reviews_vectorizer.get_feature_names())

## Using the raw token counts, let's generate a normalized version of the vector matrix
## Copy the original raw count vector matrix and add a column for the document word count
reviews_vector_norm_df = reviews_vector_df.merge(
    pd.DataFrame([{"ix": ix, "word_count": len(row["review"].split(" "))} for ix, row in reviews_df.iterrows()]).set_index("ix"),
    left_index=True, right_index=True
)
## Divide each token count by the document word count
reviews_vector_norm_df.iloc[:,:-1] = reviews_vector_norm_df.iloc[:,:-1].div(reviews_vector_norm_df["word_count"], axis=0)
## Remove the word_count column, it has server its purpose
reviews_vector_norm_df.drop(["word_count"], axis = 1, inplace=True)

## Let's generate a weighted vector matrix (TFIDF)
## The tfidf vectorizer will prune tokens if:
## - In the STOPWORDS2 collection
## - Numeric
reviews_vectorizer = TfidfVectorizer(min_df=3,
                                     stop_words=STOPWORDS2,
                                     token_pattern=r'\b[^\d\W]+\b'
                                     )
## Run the vectorizer on the review data
reviews_vector = reviews_vectorizer.fit_transform(reviews_df["review"].values)
## Replace the token ID in the column label with the token value (the actual word)
reviews_vector_tfidf_df = pd.DataFrame(reviews_vector.toarray(), columns=reviews_vectorizer.get_feature_names())

## ------------------------------------------------
## Merge the token vectors with the main review data frame (minus the original review data)
## In order to re-add the classification values
reviews_tokens_df = reviews_df.drop(["review","index"], axis=1).merge(reviews_vector_df, left_index=True, right_index=True)
reviews_tfidf_df = reviews_df.drop(["review","index"], axis=1).merge(reviews_vector_tfidf_df, left_index=True, right_index=True)
reviews_norm_df = reviews_df.drop(["review","index"], axis=1).merge(reviews_vector_norm_df, left_index=True, right_index=True)

## ---------------------------------------------
## We have two classifiers in this file, sentiment(n=negative or p=positive) and lie (t=true or f=false)
## Split the source data into six data frames, one for each classifier and vectorization type:
## - Count, normalized, & tfidf
reviews_sentiment_count_df = reviews_tokens_df.drop(["lie"], axis=1)
reviews_lie_count_df = reviews_tokens_df.drop(["sentiment"], axis=1)
reviews_sentiment_tfidf_df = reviews_tfidf_df.drop(["lie"], axis=1)
reviews_lie_tfidf_df = reviews_tfidf_df.drop(["sentiment"], axis=1)
reviews_sentiment_norm_df = reviews_norm_df.drop(["lie"], axis=1)
reviews_lie_norm_df = reviews_norm_df.drop(["sentiment"], axis=1)

## ---------------------------------------------
## WordClouds
## LET'S DO IT AGAIN!!! YOU GET A WORDCLOUD! YOU GET A WORDCLOUD!!
## * All reviews
reviews_all_freq = reviews_vector_df.sum(axis=0)
reviews_all_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(reviews_all_freq)
plt.figure()
plt.imshow(reviews_all_wc, interpolation="bilinear")
plt.axis("off")
plt.title("All Reviews Wordcloud")
plt.savefig(fname="wc_all_reviews.png", bbox_inches="tight")
##print("===================\nAll reviews term frequency")
##print(reviews_all_freq.sort_values(ascending=False)[:10])

## * Positive reviews
reviews_pos_freq = reviews_vector_df.iloc[reviews_df[reviews_df["sentiment"] == "p"].index].sum(axis=0)
reviews_pos_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(reviews_pos_freq)
plt.figure()
plt.imshow(reviews_pos_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Wordcloud")
plt.savefig(fname="wc_pos_reviews.png", bbox_inches="tight")
##print("===================\nPositive reviews term frequency")
##print(reviews_pos_freq.sort_values(ascending=False)[:10])

## * Negative reviews
reviews_neg_freq = reviews_vector_df.iloc[reviews_df[reviews_df["sentiment"] == "n"].index].sum(axis=0)
reviews_neg_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(reviews_neg_freq)
plt.figure()
plt.imshow(reviews_neg_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Wordcloud")
plt.savefig(fname="wc_neg_reviews.png", bbox_inches="tight")
##print("===================\nNegative reviews term frequency")
##print(reviews_neg_freq.sort_values(ascending=False)[:10])

## * True reviews
reviews_true_freq = reviews_vector_df.iloc[reviews_df[reviews_df["lie"] == "f"].index].sum(axis=0)
reviews_true_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(reviews_true_freq)
plt.figure()
plt.imshow(reviews_true_wc, interpolation="bilinear")
plt.axis("off")
plt.title("True Reviews Wordcloud")
plt.savefig(fname="wc_true_reviews.png", bbox_inches="tight")
##print("===================\nTrue reviews term frequency")
##print(reviews_true_freq.sort_values(ascending=False)[:10])

## * Fake reviews
reviews_fake_freq = reviews_vector_df.iloc[reviews_df[reviews_df["lie"] == "t"].index].sum(axis=0)
reviews_fake_wc = WordCloud(max_font_size=100, max_words=100, )\
   .generate_from_frequencies(reviews_fake_freq)
plt.figure()
plt.imshow(reviews_fake_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Fake Reviews Wordcloud")
plt.savefig(fname="wc_fake_reviews.png", bbox_inches="tight")
##print("===================\nFake reviews term frequency")
##print(reviews_fake_freq.sort_values(ascending=False)[:10])

## =============================================
## Part 3 - Naive Bayes modelling
## =============================================
## Generate the indexes for the folds that will be used for cross validation
fold_count = 10
## /////////////////////////////////////////////
## DISCLAIMER: This method of generating indexes for the k-folds
## validation of testing and training was authored by Professor Bolton.
## It's a clever way to insure that both classifications
## are equally represented in the training and testing folds.
## START ---------------------------------------
## Indexes for true review
reviews_true_ix = reviews_df[reviews_df["lie"] == "f"].index
## Indexes for fake review
reviews_fake_ix = reviews_df[reviews_df["lie"] == "t"].index
lie_train_ix = []
lie_test_ix = []
lie_f_train_ix = []
lie_f_test_ix = []
lie_t_train_ix = []
lie_t_test_ix = []
folds_true = KFold(n_splits=fold_count, shuffle=True)
folds_true.get_n_splits(reviews_true_ix)
for train_ix, test_ix in folds_true.split(reviews_true_ix):
    lie_f_train_ix.append(reviews_true_ix[train_ix])
    lie_f_test_ix.append(reviews_true_ix[test_ix])
folds_fake = KFold(n_splits=fold_count, shuffle=True)
folds_fake.get_n_splits(reviews_fake_ix)
for train_ix, test_ix in folds_fake.split(reviews_fake_ix):
    lie_t_train_ix.append(reviews_fake_ix[train_ix])
    lie_t_test_ix.append(reviews_fake_ix[test_ix])
for i in range(fold_count):
    lie_train_ix.append(list(lie_f_train_ix[i]) + list(lie_t_train_ix[i]))
    lie_test_ix.append(list(lie_f_test_ix[i]) + list(lie_t_test_ix[i]))
lie_kfolds_ix = list(zip(lie_train_ix,lie_test_ix))
## ---------------------------------------------
## Indexes for positive sentiment
reviews_pos_ix = reviews_df[reviews_df["sentiment"] == "p"].index
## Indexes for negative sentiment
reviews_neg_ix = reviews_df[reviews_df["sentiment"] == "n"].index
sentiment_train_ix = []
sentiment_test_ix = []
sentiment_p_train_ix = []
sentiment_p_test_ix = []
sentiment_n_train_ix = []
sentiment_n_test_ix = []
folds_pos = KFold(n_splits=fold_count, shuffle=True)
folds_pos.get_n_splits(reviews_pos_ix)
for train_ix, test_ix in folds_pos.split(reviews_pos_ix):
    sentiment_p_train_ix.append(reviews_pos_ix[train_ix])
    sentiment_p_test_ix.append(reviews_pos_ix[test_ix])
folds_neg = KFold(n_splits=fold_count, shuffle=True)
folds_neg.get_n_splits(reviews_neg_ix)
for train_ix, test_ix in folds_neg.split(reviews_neg_ix):
    sentiment_n_train_ix.append(reviews_neg_ix[train_ix])
    sentiment_n_test_ix.append(reviews_neg_ix[test_ix])
for i in range(fold_count):
    sentiment_train_ix.append(list(sentiment_p_train_ix[i]) + list(sentiment_n_train_ix[i]))
    sentiment_test_ix.append(list(sentiment_p_test_ix[i]) + list(sentiment_n_test_ix[i]))
sentiment_kfolds_ix = list(zip(sentiment_train_ix,sentiment_test_ix))
## END -----------------------------------------

"""
DISCLAIMER: The following functions come from Professor Bolton's example code
* feat_imp is designed to produce a list of the most important features\words
    to each model.
* plot_confusion_matrix will plot a viz for the confusion matrix validation
"""
## START ---------------------------------------
def feat_imp(train_df, model):
    featLogProb = []
    features = {}
    ind = 0
    for feats in train_df.columns:
        ## the following line takes the difference of the log prob of feature given model
        ## thus it measure the importance of the feature for classification.
        featLogProb.append(abs(model.feature_log_prob_[1,ind] - model.feature_log_prob_[0,ind]))
        features[(feats)] = featLogProb[ind]
        ind = ind + 1

    sortedKeys = sorted(features, key = features.get, reverse = True)[:10]
    sortedVals = sorted(features.values(), reverse = True)[:10]
    features2 = {}
    for ki in range(len(sortedKeys)):
        features2[sortedKeys[ki]] = sortedVals[ki]
    return(features2)

def plot_confusion_matrix(y_true, y_pred, classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    # ... and label them with the respective list entries
    xticklabels=classes, yticklabels=classes,
    title=title,
    ylabel='True label',
    xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
## END -----------------------------------------

reviews_matrices = {"sentiment_count": reviews_sentiment_count_df,
                    "sentiment_norm": reviews_sentiment_norm_df,
                    "sentiment_tfidf": reviews_sentiment_tfidf_df,
                    "lie_count": reviews_lie_count_df,
                    "lie_norm": reviews_lie_norm_df,
                    "lie_tfidf": reviews_lie_tfidf_df}

## ---------------------------------------------
## Run the SciKit Mulinomial model building functions on each of the vector matrices
## Record the success rates in a confusion matrix
conf_matrix_collection = {}
accuracy_collection = {}
prfs_collection = {} # Precision Recall F-Score
features_collection = {} # Holder for the most important features for each set and fold
for vm_name, vm_df in reviews_matrices.items():
    ## Prepare the containers for the validation results
    conf_matrix_collection[vm_name] = []
    accuracy_collection[vm_name] = []
    prfs_collection[vm_name] = []
    features_collection[vm_name] = []
    ## Get the kfold indeces for this classification set
    kfolds_ix = sentiment_kfolds_ix if "sentiment" in vm_name else lie_kfolds_ix
    ix = 1
    for train_ix, test_ix in kfolds_ix:
        with pd.option_context("mode.chained_assignment", None):
            ## Prepare the training data
            train_df = vm_df.iloc[train_ix,]
            train_labels = train_df["sentiment" if "sentiment" in vm_name else "lie"]
            train_df.drop(["sentiment" if "sentiment" in vm_name else "lie"], axis=1, inplace=True)
            ## Prepare the test data
            test_df = vm_df.iloc[test_ix,]
            test_labels = test_df["sentiment" if "sentiment" in vm_name else "lie"]
            test_df.drop(["sentiment" if "sentiment" in vm_name else "lie"], axis=1, inplace=True)
            ## Prepare the multinomial Naive Bayes model builder
            mnb_builder = MultinomialNB()
            ## Build the model
            mnb_builder.fit(train_df, train_labels)
            ## Use the model for prediction using the test data set
            mnb_predict = mnb_builder.predict(test_df)
            ## Validate the prediction results
            ## - Confusion matrix
            conf_matrix = confusion_matrix(test_labels.tolist(), mnb_predict.tolist(),
                                           ['n','p'] if "sentiment" in vm_name else ['t','f']
                                           )
            conf_matrix_collection[vm_name].append(conf_matrix)
            ## - Plot the confusion matrix
            title = str('Confusion Matrix\n' + vm_name + ' fold ' + str(ix))
            conf_matrix_plot = plot_confusion_matrix(y_true=test_labels.tolist(),
                                                     y_pred=mnb_predict.tolist(),
                                                     classes=['n', 'p'] if "sentiment" in vm_name else ['t', 'f'],
                                                     title=title)
            plt.savefig("figures\\cm_" + vm_name + "_fold_" + str(ix) + ".png", bbox_inches='tight')
            plt.close()
            ## - Accuracy score
            accuracy = accuracy_score(test_labels.tolist(), mnb_predict.tolist())
            accuracy_collection[vm_name].append(accuracy)
            ## - Precision recall f-score
            prfs = precision_recall_fscore_support(test_labels.tolist(), mnb_predict.tolist(),
                                                   zero_division=0, average="micro",
                                                   labels="sentiment" if "sentiment" in vm_name else "lie")
            prfs_collection[vm_name].append(prfs)
            ## - Important features for this model
            features_collection[vm_name].append(feat_imp(train_df, mnb_builder))

            ## Plot WordCloud of important words
            wc = WordCloud().generate_from_frequencies(features_collection[vm_name][ix - 1])
            plt.imshow(wc)
            plt.xticks(ticks = None)
            plt.yticks(ticks = None)
            plt.title("Important " + ("sentiment" if "sentiment" in vm_name else "lie") + " detection words - Fold " + str(ix))
            plt.savefig("figures\\wc_folds_" + vm_name + "_fold_" + str(ix) + ".png", bbox_inches='tight')
            plt.close()
            ix += 1

## ---------------------------------------------
## Summarize the results
model_summary_lie = {}
model_summary_sentiment = {}
for vm_name in reviews_matrices.keys():
    smry = {}
    ## Accuracy
    smry["accuracy"] = np.mean(accuracy_collection[vm_name][0])
    ## Precision recall f-score
    prec = 0
    for x in prfs_collection[vm_name]:
        prec += x[0]
    smry["precision"] = prec / 10
    ## Recall
    rec = 0
    for x in prfs_collection[vm_name]:
        rec += x[1]
    smry["recall"] = rec / 10
    ## F1 score
    f1 = 0
    for x in prfs_collection[vm_name]:
        f1 += x[2]
    smry["f1"] = f1 / 10
    (model_summary_sentiment if "sentiment" in vm_name else model_summary_lie)[vm_name] = smry

## Plot validation plots
for t in ["Sentiment","Lie"]:
    src_summary = model_summary_lie if t == "Lie" else model_summary_sentiment
    #Accuracy
    plt.bar(range(len(src_summary.keys())), [float(x["accuracy"]) for x in src_summary.values()], align="center")
    plt.xticks(range(len(src_summary.keys())), list(src_summary.keys()))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title("Average Accuracy Over 10 Folds Predicting " + t)
    plt.savefig("figures\\" + t.lower() + "_model_accuracy.png", bbox_inches='tight')
    plt.close()
    
    #Precision
    plt.bar(range(len(src_summary.keys())), [float(x["precision"]) for x in src_summary.values()], align="center")
    plt.xticks(range(len(src_summary.keys())), list(src_summary.keys()))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title("Average Precision Over 10 Folds Predicting " + t)
    plt.savefig("figures\\" + t.lower() + "_model_precision.png", bbox_inches='tight')
    plt.close()
    
    #Recall
    plt.bar(range(len(src_summary.keys())), [float(x["recall"]) for x in src_summary.values()], align="center")
    plt.xticks(range(len(src_summary.keys())), list(src_summary.keys()))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title("Average Recall Over 10 Folds Predicting " + t)
    plt.savefig("figures\\" + t.lower() + "_model_recall.png", bbox_inches='tight')
    plt.close()
    
    #F1
    plt.bar(range(len(src_summary.keys())), [float(x["f1"]) for x in src_summary.values()], align="center")
    plt.xticks(range(len(src_summary.keys())), list(src_summary.keys()))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title("Average F1 Over 10 Folds Predicting " + t)
    plt.savefig("figures\\" + t.lower() + "_model_f1.png", bbox_inches='tight')
    plt.close()

