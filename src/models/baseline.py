import re
from src.reader.pan_hatespeech import prepare_for_pan_baseline
from emoji import UNICODE_EMOJI
from pathlib import Path
from src.utils import data_args
from argparse import ArgumentParser
from src.reader.pan_hatespeech import prepare_for_pan_baseline
import pickle
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

'''
This is implementation of last year PAN winner
Credits: https://github.com/pan-webis-de/buda20/blob/main/final_software/testing_script.py
'''


# text cleaning v1
def cleaning_v1(tweet_lista):
    cleaned_feed_v1 = []
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        cleaned_feed_v1.append(feed)
    return cleaned_feed_v1


# emoji handling
def is_emoji(s):
    return s in UNICODE_EMOJI


def emoji_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def cleaning_v2(tweet_lista):
    cleaned_feed_v2 = []
    for feed in tweet_lista:
        feed = feed.lower()
        feed = emoji_space(feed)
        feed = re.sub('[,.\'\"\‘\’\”\“]', '', feed)
        feed = re.sub(r'([a-z\'-\’]+)', r'\1 ', feed)
        feed = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )', '', feed)
        feed = re.sub('[\n]', ' ', feed)
        feed = ' '.join(feed.split())
        cleaned_feed_v2.append(feed)
    return cleaned_feed_v2


class Model:
    def __init__(self, args):
        self.model_output_dir = Path(args.model_output_dir)
        self.lang = args.lang

    def train(self, data):
        # clean English data
        feed_list = data['inputs']
        targets = data['targets']
        cleaned_texts_1 = cleaning_v1(feed_list)
        cleaned_texts_2 = cleaning_v2(feed_list)

        lr_vectorizer_path = self.model_output_dir / self.lang / 'lr_vectorizer.pickle'
        if not lr_vectorizer_path:
            lr_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=6, sublinear_tf=True, use_idf=True,
                                            smooth_idf=True)
            lr_vectorizer.fit_transform(cleaned_texts_1)
            pickle.dump(lr_vectorizer,
                        open(lr_vectorizer_path, 'wb'))
        else:
            lr_vectorizer = pickle.load(open(lr_vectorizer_path, 'rb'))
            X_LR = lr_vectorizer.transform(cleaned_texts_1)

        svm_vectorizer_path = self.model_output_dir / self.lang / 'svm_vectorizer.pickle'
        if not svm_vectorizer_path:
            svm_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, use_idf=True,
                                             smooth_idf=True)
            svm_vectorizer.fit_transform(cleaned_texts_1)
            pickle.dump(svm_vectorizer, open(svm_vectorizer_path, 'wb'))
        else:
            svm_vectorizer = pickle.load(open(svm_vectorizer_path, 'rb'))
            X_SVM = svm_vectorizer.transform(cleaned_texts_1)

        rf_vectorizer_path = self.model_output_dir / self.lang / 'rf_vectorizer.pickle'
        if not rf_vectorizer_path:
            rf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=9)
            rf_vectorizer.fit_transform(cleaned_texts_1)
            pickle.dump(rf_vectorizer, open(rf_vectorizer_path, 'wb'))
        else:
            rf_vectorizer = pickle.load(open(rf_vectorizer_path, 'rb'))
            X_rf = rf_vectorizer.transform(cleaned_texts_1)

        xgb_vectorizer_path = self.model_output_dir / self.lang / 'xgb_vectorizer.pickle'
        if not xgb_vectorizer_path:
            xgb_vectorizer = TfidfVectorizer(min_df=8, ngram_range=(1, 2), use_idf=True, smooth_idf=True,
                                             sublinear_tf=True)
            xgb_vectorizer.fit_transform(cleaned_texts_1)
            pickle.dump(xgb_vectorizer, open(xgb_vectorizer_path, 'wb'))
        else:
            xgb_vectorizer = pickle.load(open(xgb_vectorizer_path, 'rb'))
            X_xgb = xgb_vectorizer.transform(cleaned_texts_1)

        # Models
        lr_model_path = self.model_output_dir / self.lang / 'lr_model.sav'
        if not lr_model_path:
            lr_model = LogisticRegression(C=1000, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0,
                                          random_state=5)
            lr_model.fit(X_LR, targets)
            pickle.dump(lr_model, open(lr_model_path, 'wb'))

        svm_model_path = self.model_output_dir / self.lang / 'svm_model.sav'
        if not svm_model_path:
            svm_model = svm.SVC(C=100, kernel='linear', probability=True, verbose=False)
            svm_model.fit(X_SVM, targets)
            pickle.dump(svm_model, open(svm_model_path, 'wb'))

        rf_model_path = self.model_output_dir / self.lang / 'rf_model.sav'
        if not rf_model_path:
            rf_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=9, criterion='gini', random_state=0,
                                              oob_score=True)
            rf_model.fit(X_rf, targets)
            pickle.dump(rf_model, open(rf_model_path, 'wb'))

        xgb_model_path = self.model_output_dir / self.lang / 'xgb_model.sav'
        if not xgb_model_path:
            xgb_model = xgb.XGBClassifier(colsample_bytree=0.6, eta=0.01, max_depth=6, n_estimators=300, subsample=0.8)
            xgb_model.fit(X_xgb, targets)
            pickle.dump(xgb_model, open(xgb_model_path))

        # predicting
        preds_rf = rf_model.predict_proba(X_rf)[:, 1]
        preds_svm = svm_model.predict_proba(X_SVM)[:, 1]
        preds_LR = lr_model.predict_proba(X_LR)[:, 1]
        preds_XGB = xgb_model.predict_proba(X_xgb)[:, 1]

        ##
        ## feature extraction
        ##

        # initialize feature DF with ID-s per author
        # en_data_tweet_consist = pd.DataFrame(list(zip([en_data_tw["ID"][i * 100] for
        #                                                i in range(int(len(en_data_tw["ID"]) / 100))])), columns=['ID'])

        ## length stats of tweets per author (charachter & word)
        # len_tw_char = [len(i) for i in en_data_tw["Tweets"]]
        # len_tw_word = [len(i.split(" ")) for i in en_data_tw["Tweets"]]

        # # SD
        # len_char_sd_auth = [pstdev(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
        # len_word_sd_auth = [pstdev(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]
        #
        # # min - max - range - mean
        # len_char_min_auth = [min(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
        # len_word_min_auth = [min(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]
        #
        # len_char_max_auth = [max(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
        # len_word_max_auth = [max(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]
        #
        # len_char_rng_auth = [max(len_tw_char[i * 100:i * 100 + 99]) - min(len_tw_char[i * 100:i * 100 + 99]) for
        #                      i in range(int(len(len_tw_char) / 100))]
        # len_word_rng_auth = [max(len_tw_word[i * 100:i * 100 + 99]) - min(len_tw_word[i * 100:i * 100 + 99]) for
        #                      i in range(int(len(len_tw_word) / 100))]
        #
        # len_char_mean_auth = [np.mean(len_tw_char[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_char) / 100))]
        # len_word_mean_auth = [np.mean(len_tw_word[i * 100:i * 100 + 99]) for i in range(int(len(len_tw_word) / 100))]
        #
        # ##vocab variety
        # tweets_szerz = [" ".join(list(en_data_tw["Tweets"])[i * 100:99 + i * 100]) for
        #                 i in range(int(len(len_tw_char) / 100))]  # concate tweets of each author to calculate TTR
        #
        # ttr_szerz = [ld.ttr(ld.flemmatize(i)) for i in tweets_szerz]
        #
        # ##number of tags
        # # RT
        # rt_szerz = [np.sum([k == "RT" for k in i.split(" ")]) for i in tweets_szerz]
        # # URL
        # url_szerz = [np.sum([k == "#URL#" for k in i.split(" ")]) for i in tweets_szerz]
        # # hashtag
        # hsg_szerz = [np.sum([k == "#HASHTAG#" for k in i.split(" ")]) for i in tweets_szerz]
        # # user
        # user_szerz = [np.sum([k == "#USER#" for k in i.split(" ")]) for i in tweets_szerz]
        # # ...
        # p_szerz = [np.sum([k[-1:] == "…" for k in i.split(" ")]) for i in tweets_szerz]
        # # emoji
        # emoj_szerz = []
        # for aut in tweets_szerz:
        #     emdb = 0
        #     for tok in aut.split(" "):
        #         for c in tok:
        #             emdb += c in UNICODE_EMOJI
        #     emoj_szerz.append(emdb)
        #
        # # populate feature DF
        # en_data_tweet_consist["len_char_sd_auth"] = len_char_sd_auth
        # en_data_tweet_consist["len_word_sd_auth"] = len_word_sd_auth
        #
        # en_data_tweet_consist["len_char_min_auth"] = len_char_min_auth
        # en_data_tweet_consist["len_word_min_auth"] = len_word_min_auth
        #
        # en_data_tweet_consist["len_char_max_auth"] = len_char_max_auth
        # en_data_tweet_consist["len_word_max_auth"] = len_word_max_auth
        #
        # en_data_tweet_consist["len_char_rng_auth"] = len_char_rng_auth
        # en_data_tweet_consist["len_word_rng_auth"] = len_word_rng_auth
        #
        # en_data_tweet_consist["len_char_mean_auth"] = len_char_mean_auth
        # en_data_tweet_consist["len_word_mean_auth"] = len_word_mean_auth
        #
        # en_data_tweet_consist["rt_szerz"] = rt_szerz
        # en_data_tweet_consist["url_szerz"] = url_szerz
        # en_data_tweet_consist["hsg_szerz"] = hsg_szerz
        # en_data_tweet_consist["user_szerz"] = user_szerz
        # en_data_tweet_consist["p_szerz"] = p_szerz
        # en_data_tweet_consist["emoj_szerz"] = emoj_szerz
        # en_data_tweet_consist["ttr_szerz"] = ttr_szerz

    def eval(self, path):
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', help='Enter the directory to save the trained models')
    parser = data_args(parser)
    args = parser.parse_args()
    model = Model(args)
    data = prepare_for_pan_baseline(args.data_path, args.lang)
    model.train()
