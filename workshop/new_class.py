import os
import re
import fnmatch
import pandas as pd
import numpy as np
# install the package beforehand
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
# choose your model below
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier


class classifier_workshop():

    # return a pd dataframe with path and label
    def get_files(self, tmp_path):
        the_files = pd.DataFrame()
        for root, dir, files in os.walk(tmp_path):
            for items in fnmatch.filter(files, "*"):
                tmp = root.split("/")
                tmp = tmp[len(tmp)-1] # extract the label
                the_files = the_files.append({"path":root+"/"+items, "label": tmp}, ignore_index=True)
        return the_files

    # return a dataframe with text and label
    def clean_text(self, df):
        stopWords = set(stopwords.words("english"))
        the_new = pd.DataFrame()

        # zip a tuple
        for file_name, the_label in zip(df["path"], df["label"]):
            f = open(file_name, "r", encoding = "utf-8")

            tmp_text = f.readlines()
            # join with space
            tmp_text = " ".join(tmp_text)

            the_text = tmp_text.split()
            the_text = [word.lower() for word in the_text]

            # get rid of special leters, "^" is a negation sign
            the_text = [re.sub(r'[^a-zA-Z0-9]+', " ", token) for token in the_text]
            # drop "\s+" 
            the_text = [word for word in the_text if len(re.sub(r"\s+", "", word)) != 0]
            # drop stop words
            the_text = [word for word in the_text if word not in stopWords]
            the_text = " ".join(the_text)
            the_text = re.sub(" +", " ", the_text)

            the_new = the_new.append({"body": the_text, "label": the_label}, ignore_index = True)
            f.close()
            
        return the_new

    # train model
    def the_train(self, model, df):
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range = (1, 3))
        the_vec = vectorizer.fit_transform(df.body).toarray()

        tdm = pd.DataFrame(the_vec)
        # setting column names
        tdm.columns = vectorizer.get_feature_names() 

        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(df[df.columns[1]])
        the_labels = [label_enc.transform([word])[0] for word in df[df.columns[1]]]

        #the_score = cross_val_score(model, tdm, the_labels, cv = 10)
        #mean_score = np.mean(the_score)

        model.fit(tdm, the_labels)

        return vectorizer, model, label_enc

    # make prediction
    def the_classify(self, v, m, l, to_predict):
        tdm_pred = v.transform(to_predict)
        prediction = m.predict(tdm_pred)

        the_prob = pd.DataFrame(m.predict_proba(tdm_pred))
        the_prob.columns = l.classes_

        the_pred = l.inverse_transform([prediction][0])

        return the_pred


if __name__ == "__main__":
    # part 1
    the_files = pd.DataFrame()
    # download the data and set your own path
    path = "/Users/apple/Desktop/semester_1/5.nlp/workshop/workshop_data"

    ws = classifier_workshop()
    the_files = ws.get_files(path)
    # you may not need this line
    the_files = the_files.drop([0])

    # part 2
    stopWords = set(stopwords.words("english"))
    the_new = ws.clean_text(the_files)

    # part 3
    # return vectorizer, model, label_enc
    v, m, l = ws.the_train(GradientBoostingClassifier(), the_new)

    # part 4
    # change your input here
    to_predict = ["this expalins the universe through numbers"]
    the_pred = ws.the_classify(v, m, l, to_predict)

    # it takes 30 seconds also to run the model on my macbook pro
    print(the_pred)


'''
conclusion

This is a simple introduction to nlp. It includes how to preprocess text data and how to call those state-of-the-art models, 
like random forest and ada boost, from sklearn. Unfortunately, it doesn't include the principles or mathematics behind the models 
due to time limit. And of course, it is impossible to finish these in just one day. Also, it doesn't include how to use these models 
or how to choose the best model given a specific problem. Finally, when we applied the random forest model using twitter api, it seemed 
a little biased. In my view, it might be because of the problem of overfitting and the limit of the training data. Therefore, in order 
to delve deeper into the field of nlp and build a better model, it might be helpful to think more like a linguistist instead of a 
statistician or computer scientist. In other words, we shouldn't just simply calculate the probability of the words or grams. We need 
to take the relationship between the words and words or words and sentences into consideration, just like the pcfg model, cky parser and 
dependency parser.

Here are several questions in my mind:
1. what is the difference between nlp taught in university and nlp applied in the industrial field;
2. what is the cutting-edge of nlp today;
3. what is the difficulty that researchers and scientists encountered nowadays.
'''




