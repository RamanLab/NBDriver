#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shayantan Banerjee
path = "/data/shayantan/NBDriver/python/data"
This program performs the repeated 10-fold cross-validation experiments using the KDE classifier for the Brown et al. dataset
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import scipy as scp
import glob
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import arange
from numpy import argmax
from sklearn import metrics
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def pd_read_pattern():
    #Loads all the required files into a list
    file_paths_pos = glob.glob(path+"/*pos.txt")
    pos_data=[pd.read_csv(f,sep="\t") for f in file_paths_pos]
    file_paths_kmer = glob.glob(path+"/*kmer.txt")
    kmer_data=[pd.read_csv(f,sep="\t") for f in file_paths_kmer]
    return pos_data,kmer_data

def shuffle_data(list_of_dataset):
    new_list=[]
    for i in range(0,len(list_of_dataset)):
        shuff_data=shuffle(list_of_dataset[i])
        shuff_data.reset_index(inplace=True, drop=True)
        new_list.append(shuff_data)
    return new_list

def convert_to_labels(probs, thresh):
    return (probs >= thresh).astype('int')

from sklearn.base import BaseEstimator, ClassifierMixin


class KDEClassifier(BaseEstimator, ClassifierMixin):
    #    KDE classifier implementation from the Python Data Science Handbook by Jake VanderPlas
    """Bayesian generative classification based on KDE
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scp.stats.sem(a)
    h = se * scp.stats.t.ppf((1 + confidence) / 2., n-1)
    print(m, " ", "(",m-h,"," ,m+h,")")
    
def preprocess(data,k):
    copy = [[x[i:i+k] for i in range(len(x)-k+1)] for x in data]
    copy=[" ".join(review) for review in copy]
    return copy
def dat_prep(nbd_train,nbd_test,k,vect_type,Type_train,Type_test,Chr_train,Chr_test,Label_train,Label_test):
    if(vect_type=="CV"):
        vect=Pipeline([('cv1',CountVectorizer(lowercase=False))])
    else:
        vect = Pipeline([('cv1',CountVectorizer(lowercase=False)), ('tfidf_transformer',TfidfTransformer(smooth_idf=True,use_idf=True))])

    count_vector_train=vect.fit_transform(preprocess(nbd_train,k))
    count_vector_test=vect.transform(preprocess(nbd_test,k))

    df_train=pd.DataFrame(count_vector_train.todense(),columns=vect['cv1'].get_feature_names())
    df_train['Type']=Type_train
    df_train['Label']=Label_train
    df_train['Chr']=Chr_train
    df_test=pd.DataFrame(count_vector_test.todense(),columns=vect['cv1'].get_feature_names())
    df_test['Type']=Type_test
    df_test['Label']=Label_test
    df_test['Chr']=Chr_test
    
    
    return df_train,df_test       
def cv_tf_transformation(df_train,df_test):
    f=0
    if(len(df_train['new_nbd'].tolist()[0])==3):
        df_2_cv_train,df_2_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_3_cv_train,df_3_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_2_tf_train,df_2_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_3_tf_train,df_3_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        f=1
        
    else:
        df_2_cv_train,df_2_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_3_cv_train,df_3_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_4_cv_train,df_4_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),4,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_2_tf_train,df_2_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_3_tf_train,df_3_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())
        df_4_tf_train,df_4_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),4,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist())

    if f==1:
        names1=['df_2_cv','df_3_cv','df_2_tf','df_3_tf']
        dat_train_1=[df_2_cv_train, df_3_cv_train, df_2_tf_train, df_3_tf_train]
        dat_test_1=[df_2_cv_test,df_3_cv_test,df_2_tf_test,df_3_tf_test]
        return dat_train_1,dat_test_1,names1
    else:
        names2=['df_2_cv','df_3_cv','df_4_cv','df_2_tf','df_3_tf','df_4_tf']
        dat_train_2=[df_2_cv_train, df_3_cv_train, df_4_cv_train, df_2_tf_train, df_3_tf_train, df_4_tf_train]
        dat_test_2=[df_2_cv_test,df_3_cv_test,df_4_cv_test, df_2_tf_test,df_3_tf_test,df_4_tf_test]

        return dat_train_2,dat_test_2,names2
def feature_reduction_using_trees(X,y):
    np.random.seed(315)
    forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=42)

    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #print(importances[indices])
    n=0.3*len(X.columns)
    feat_list=X.columns[indices][0:int(n)]
    return X[feat_list]

def run_repeatedCV():
    sen_df=pd.DataFrame();spe_df=pd.DataFrame();auc_df=pd.DataFrame();mcc_df=pd.DataFrame();k=0
    pos_data,kmer_data=pd_read_pattern()
    new_kmer=shuffle_data(kmer_data)
    #KDE kmer using top 30 percentile features with tfidf scores for test derived from train using transform()
    for i in range(0,10):
        ctr=0
        print("Window size: ",i+1,"\n")
        if(i==0):
            names2=['df_2_cv','df_3_cv','df_2_tf','df_3_tf']
        else:
            names2=['df_2_cv','df_3_cv','df_4_cv','df_2_tf','df_3_tf','df_4_tf']
        X=new_kmer[i];y=new_kmer[i]['Label'];
        print("kmer size", names2[ctr])
        ctr=ctr+1
        rskf=RepeatedStratifiedKFold(n_splits=10,n_repeats=3);
        for train_index,test_index in rskf.split(X,y):
            sen_kde=[];spe_kde=[];acc_kde=[];auc_kde=[];m_kde=[];c=[];
            k=k+1
            print(k,end=",")
            X_train,X_test=X.iloc[train_index],X.iloc[test_index]
            dat_train,dat_test,names=cv_tf_transformation(X_train,X_test)
    
            for j in range(0,len(dat_train)):
                dat_train[j]['Chr'] = dat_train[j]['Chr'].replace(['X'],'21')
                dat_test[j]['Chr']=dat_test[j]['Chr'].replace(['X'],'21')
                train_x=dat_train[j].drop('Label',axis=1);train_y=dat_train[j]['Label'];    
                test_x=dat_test[j].drop('Label',axis=1);test_y=dat_test[j]['Label'];
                X_red=feature_reduction_using_trees(train_x,train_y)
                bandwidths = np.logspace(-1, 1, 30)
                grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths},cv=3)
                grid.fit(X_red,train_y)
                best_model=grid.best_estimator_
                best_model.fit(X_red,train_y)
                y_probs = best_model.predict_proba(test_x[X_red.columns])[:,1]
                thresholds = arange(0, 1, 0.001)
                scores = [roc_auc_score(test_y, convert_to_labels(y_probs, t)) for t in thresholds]
                ix= argmax(scores)
                y_test_predictions = np.where(best_model.predict_proba(test_x[X_red.columns])[:,1] > thresholds[ix], 2, 1)
                sensi= sensitivity_score(test_y, y_test_predictions, pos_label=2)
                speci=specificity_score(test_y,y_test_predictions,pos_label=2)
                accu=accuracy_score(test_y,y_test_predictions)
                auro=roc_auc_score(test_y,y_test_predictions)
                mcc=metrics.matthews_corrcoef(test_y,y_test_predictions)
                c.append(X_red.columns)
                sen_kde.append(sensi);spe_kde.append(speci);acc_kde.append(accu);auc_kde.append(auro);m_kde.append(mcc)
            if(i==0):
                sen_df = sen_df.append({'df_2_cv': sen_kde[0], 'df_3_cv': sen_kde[1], 'df_2_tf': sen_kde[2],'df_3_tf':sen_kde[3]}, ignore_index=True)
                spe_df = spe_df.append({'df_2_cv': spe_kde[0], 'df_3_cv': spe_kde[1], 'df_2_tf': spe_kde[2],'df_3_tf':spe_kde[3]}, ignore_index=True)
                auc_df = auc_df.append({'df_2_cv': auc_kde[0], 'df_3_cv': auc_kde[1], 'df_2_tf': auc_kde[2],'df_3_tf':auc_kde[3]}, ignore_index=True)
                mcc_df = mcc_df.append({'df_2_cv': m_kde[0], 'df_3_cv': m_kde[1], 'df_2_tf': m_kde[2],'df_3_tf':m_kde[3]}, ignore_index=True)
            else:
                sen_df = sen_df.append({'df_2_cv': sen_kde[0], 'df_3_cv': sen_kde[1],'df_4_cv':sen_kde[2], 'df_2_tf': sen_kde[3],'df_3_tf':sen_kde[4],'df_4_tf':sen_kde[5]}, ignore_index=True)
                spe_df = spe_df.append({'df_2_cv': spe_kde[0], 'df_3_cv': spe_kde[1],'df_4_cv':spe_kde[2], 'df_2_tf': spe_kde[3],'df_3_tf':spe_kde[4],'df_4_tf':spe_kde[5]}, ignore_index=True)
                auc_df = auc_df.append({'df_2_cv': auc_kde[0], 'df_3_cv': auc_kde[1],'df_4_cv':auc_kde[2], 'df_2_tf': auc_kde[3],'df_3_tf':auc_kde[4],'df_4_tf':auc_kde[5]}, ignore_index=True)
                mcc_df = mcc_df.append({'df_2_cv': m_kde[0], 'df_3_cv': m_kde[1],'df_4_cv':m_kde[2], 'df_2_tf': m_kde[3],'df_3_tf':m_kde[4],'df_4_tf':m_kde[5]}, ignore_index=True)
        #Results of this analysis or the individual dataframes sen_df,spe_df,auc_df and mcc_df for each window size and feature representation is presnt the folder titled results/KMER_KDE_new
def main():
    print("Calculating the KDE classifier's performance through repeated CV")
    run_repeatedCV()

if __name__ == "__main__":
      main()
                   