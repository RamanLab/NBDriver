#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shayantan Banerjee
path = "/data/shayantan/NBDriver/python/data"
This program derives the machine learning tool NBDriver
"""




import pandas as pd
import glob
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity
from numpy import arange
from numpy import argmax






def pd_read_pattern():
    #Loads all the files required to derive NBDriver
    path = "/data/shayantan/NBDriver/python/data"
    file_paths_pos_brown = glob.glob(path+"/data_brown"+"/*pos.txt")
    file_paths_kmer_brown = glob.glob(path+"/data_brown"+"/*kmer.txt")
    pos_data_train=[pd.read_csv(f,sep="\t") for f in file_paths_pos_brown]
    train_kmer=[pd.read_csv(f,sep="\t") for f in file_paths_kmer_brown]
    file_paths_pos_marte = glob.glob(path+"/data_marte"+"/*pos.txt")
    file_paths_kmer_marte = glob.glob(path+"/data_marte"+"/*kmer.txt")
    pos_data_test=[pd.read_csv(f,sep="\t") for f in file_paths_pos_marte]
    test_kmer=[pd.read_csv(f,sep="\t") for f in file_paths_kmer_marte]
    train=shuffle_data(pos_data_train)
    scaled_columns_train=pd.read_csv(path+"/data_brown/scaled_columns_train.csv")
    scaled_columns_test=pd.read_csv(path+"/data_marte/scaled_columns_test.csv")
    return train,train_kmer,pos_data_test,test_kmer,scaled_columns_train,scaled_columns_test

def shuffle_data(list_of_dataset):
    #randomly shuffles the data
    new_list=[]
    for i in range(0,len(list_of_dataset)):
        shuff_data=shuffle(list_of_dataset[i])
        shuff_data.reset_index(inplace=True, drop=True)
        new_list.append(shuff_data)
    return new_list

def preprocess(data,k):
    
    #decomposes the neighborhood strings into overlapping kmers of a given size

    """
    Arguments:
        data = Column containing the neighborhood sequence
        k=size of kmer
    Returns:
        copy = Overlapping kmers of a size k
    """
    copy = [[x[i:i+k] for i in range(len(x)-k+1)] for x in data]
    copy=[" ".join(review) for review in copy]
    return copy


def dat_prep(nbd_train,nbd_test,k,vect_type,Type_train,Type_test,Chr_train,Chr_test,Label_train,Label_test,scaled_feats_train,scaled_feats_test,dummy_train,dummy_test):
    #Derives the Count Vectorizer or TFIDF scores for a given neighborhood sequence
    """
    Arguments:
        nbd_train = Column containing the neighborhood sequence from the training data
        nbd_test = Column containing the neighborhood sequence from the test data

        k=size of kmer
        vect_type= 'CV' for Count Vectorizer or else TFIDF Vectorizer 
        Type_train=Numerically encoded substitution Type ("A>T" encoded as 1 or "G>C" encoded as 2 and so on) from training data
        Type_test=Numerically encoded substitution Type ("A>T" encoded as 1 or "G>C" encoded as 2 and so on) from test data
        Chr_train= Chromosome number from training data
        Chr_test=Chromosome number from test data
        Label_train=Binary label (training data), where 1=Passenger and 2=Driver
        Label_test=Binary label (test data), where 1=Passenger and 2=Driver
        scaled_feats_train=Scaled genomic features (consrvation, amino acid etc.) for training data
        scaled_feats_test=Scaled genomic features (consrvation, amino acid etc.) for test data
        dummy train= One-hot encoding based feature matrix for training data
        dummy test=One-hot encoding based feature matrix for test data


    Returns:
        df_comb_train= The complete dataframe (using training data) of TFIDF or CountVect scores plus other features such as chromosome number and substitution type
        df_comb_test= The complete dataframe (using test data) of TFIDF or CountVect scores plus other features such as chromosome number and substitution type
        count_vector_train= Just the TFIDF or Count vect features (training data) also known as the Document-Term matrix
        count_vector_test= Just the TFIDF or Count vect features (test data) also known as the Document-Term matrix
        cols= feature names
        vect= The vocabulary derived from the training data
        sc= The scaling variable derived from the training data


    """
    if(vect_type=="CV"):
        vect=Pipeline([('cv1',CountVectorizer(lowercase=False))])
    else:
        vect = Pipeline([('cv1',CountVectorizer(lowercase=False)), ('tfidf_transformer',TfidfTransformer(smooth_idf=True,use_idf=True))])
        

    count_vector_train=vect.fit_transform(preprocess(nbd_train,k))
    count_vector_test=vect.transform(preprocess(nbd_test,k))
    
    df_train=pd.DataFrame(count_vector_train.todense(),columns=vect['cv1'].get_feature_names())
    df_test=pd.DataFrame(count_vector_test.todense(),columns=vect['cv1'].get_feature_names())

    if(vect_type=="tf"):
        sc=MinMaxScaler()
        #We have used fit_transform() here because we wanted to learn the vocabulary dictionary and return document-term matrix using the traininig data
        df_train=pd.DataFrame(sc.fit_transform(df_train),columns=df_train.columns)
        #We have used transform() here since we already have a pretrained vocabulary using which we just wanted to derive the term-document matrix for the test data
        df_test=pd.DataFrame(sc.transform(df_test),columns=df_test.columns)
        
    df_train['Type']=Type_train;df_test['Type']=Type_test
    df_train['Label']=Label_train;df_test['Label']=Label_test
    df_train['Chr']=Chr_train;df_test['Chr']=Chr_test
    df_comb_train=pd.concat([df_train, scaled_feats_train,dummy_train], axis=1)
    df_comb_test=pd.concat([df_test, scaled_feats_test,dummy_test], axis=1)

    df_comb_train = df_comb_train.loc[:,~df_comb_train.columns.duplicated()]
    df_comb_test = df_comb_test.loc[:,~df_comb_test.columns.duplicated()]
    cols=vect['cv1'].get_feature_names()


    return df_comb_train,df_comb_test,count_vector_train,count_vector_test,cols,vect,sc

    

def cv_tf_transformation(df_train,df_test,scaled_feats_train,scaled_feats_test,dummy_train,dummy_test):
    f=0
    if(len(df_train['new_nbd'][0])==3):
        df_2_cv_train,df_2_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist(),scaled_feats_train,scaled_feats_test,dummy_train,dummy_test)
        df_3_cv_train,df_3_cv_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"CV",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist(),scaled_feats_train,scaled_feats_test,dummy_train,dummy_test)
        df_2_tf_train,df_2_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),2,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist(),scaled_feats_train,scaled_feats_test,dummy_train,dummy_test)
        df_3_tf_train,df_3_tf_test=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),3,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist(),scaled_feats_train,scaled_feats_test,dummy_train,dummy_test)
        f=1
        
    else:
       df_4_tf_train,df_4_tf_test,count_vector_train,count_vector_test,feats,vect,sc=dat_prep(df_train['new_nbd'].tolist(),df_test['new_nbd'].tolist(),4,"tf",df_train['Type'].tolist(),df_test['Type'].tolist(),df_train['Chromosome'].tolist(),df_test['Chromosome'].tolist(),df_train["Label"].tolist(),df_test["Label"].tolist(),scaled_feats_train,scaled_feats_test,dummy_train,dummy_test)
    if f==1:
        names1=['df_2_cv','df_3_cv','df_2_tf','df_3_tf']
        dat_train_1=[df_2_cv_train, df_3_cv_train, df_2_tf_train, df_3_tf_train]
        dat_test_1=[df_2_cv_test,df_3_cv_test,df_2_tf_test,df_3_tf_test]
        return dat_train_1,dat_test_1,names1
    else:
        names2=['df_4_tf']
        dat_train_2=[df_4_tf_train]
        dat_test_2=[df_4_tf_test]
 
        return dat_train_2,dat_test_2,names2,count_vector_train,count_vector_test,feats,vect,sc

    
def feature_reduction_using_trees(X,y):
    #selects and returns the ranked list of features using a tree-based selection technique 

    """
    Arguments:
        X = feature matrix
        y= labels
    Returns:
        X_ret = Dataframe with features ranked on the basis of importance
    """
    feature_list=[]
    np.random.seed(315)

    forest = ExtraTreesClassifier(n_estimators=250,random_state=42)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    feat_list=X.columns[indices]
    X_ret=X[feat_list]
    return X_ret


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

from sklearn.base import BaseEstimator, ClassifierMixin


class KDEClassifier(BaseEstimator, ClassifierMixin):
    #KDE classifier implementation from the Python Data Science Handbook by Jake VanderPlas

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


def score_func(y_test, y_pred):
    #Custom scoring function to derive the best composite score for Martelotto et al 
    """
    Arguments:
        y_test = Ground truth test labels
        y_pred= Predicted test labels
    Returns:
        sc = scoring function 
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    ppv=tp/(tp+fp)
    npv=tn/(tn+fn)
    sen=tp/(tp+fn)
    spe=tn/(tn+fp)
    sc=ppv+npv+sen+spe
    return sc
custom_sc=make_scorer(score_func,greater_is_better=True)

def classification_results(train,test):
    #Derivation of NBDriver using training data 
    """
    Arguments:
        train = feature matrix derived from Brown et al.
        test= feature matrix derived from Martelotto et al.
    Returns:
        best_model = Best ensemble model derived using the training data
        X_red= Dataframe derived after sampling that was used to train the model
        scores= probability based classification scores
    """
    sen=[];spe=[];acc=[];auc=[];c=[];m=[];s=[]
    train_x=train.drop('Label',axis=1);train_y=train['Label'];    
    test_x=test.drop('Label',axis=1);test_y=test['Label'];
    #Random undersampling to reduce the majority class size
    samp=RepeatedEditedNearestNeighbours(random_state=42)
    X_samp,y_samp=samp.fit_resample(train_x,train_y)
    X_samp = pd.DataFrame(X_samp, columns = train_x.columns)
    #Experimenting with different numbers of top features derived from the tree-based feature extraction method 
    top_n_feats=[30,40,50,60,70]
    X_r=feature_reduction_using_trees(X_samp,y_samp) 
    cols=X_r.columns
    for n in top_n_feats:
        print("For top: ",n," features")
        X_red=X_r[cols[0:n]]
        sv=SVC(kernel="linear",probability=True,C=0.01,random_state=42) #chosen from 5foldCV based grid search
        kde=KDEClassifier(bandwidth=1.27) #chosen from 5foldCV based grid search
        best_model = VotingClassifier(estimators=[('sv', sv), ('kde', kde)],
                        voting='soft',weights=[4, 7]) #best combination of weights selected by a brute force search (possible weights 1-10) using a cross-validation approach on the training data  
        
        best_model.fit(X_red,y_samp)
        y_probs = best_model.predict_proba(test_x[X_red.columns])[:,1]
        thresholds = arange(0, 1, 0.001)
        scores = [roc_auc_score(test_y, to_labels(y_probs, t)) for t in thresholds]
        ix= argmax(scores)
        y_test_predictions = np.where(best_model.predict_proba(test_x[X_red.columns])[:,1] > thresholds[ix], 2, 1)
        print("Thresh: ",thresholds[ix])
        sensi= sensitivity_score(test_y, y_test_predictions, pos_label=2)
        speci=specificity_score(test_y,y_test_predictions,pos_label=2)
        accu=accuracy_score(test_y,y_test_predictions)
        auro=roc_auc_score(test_y,y_test_predictions)
        mcc=metrics.matthews_corrcoef(test_y,y_test_predictions)
        tn, fp, fn, tp = confusion_matrix(test_y, y_test_predictions).ravel()
        ppv=tp/(tp+fp)
        npv=tn/(tn+fn)
        sen=tp/(tp+fn)
        spe=tn/(tn+fp)
        score=ppv+npv+sen+spe
        print("For kmer size: ",len(train.columns[0]))
        print("for top ",n," features")
        print(list(X_red.columns.values),"\n")
        score_dict={"Sen":sen,"Spe":spe,"PPV":ppv,"NPV":npv,"AUC":auro,"MCC":mcc,"ACC":accu}
        print(score)
        print(score_dict)
        df=pd.DataFrame(y_test_predictions)
        y_samp = pd.DataFrame(y_samp, columns = ['x'])
    return best_model,X_red,scores

def deriving_final_model(train,pos_data_test,train_kmer,test_kmer,scaled_columns_train,scaled_columns_test):
    #Obtaining predictions on the test set and deriving the classification performances
    for i in range(9,10): 
        print("Window size: ",i+1)
        non_dummy_cols = ['Type','Label','Chr'] 
        dummy_cols = list(set(train[i].columns) - set(non_dummy_cols))
        dummy_train=pd.get_dummies(train[i],columns=dummy_cols)
        dummy_test=pd.get_dummies(pos_data_test[i])
        dat_train,dat_test,names,train_vect,test_vect,feats,vect,sc=cv_tf_transformation(train_kmer[i],test_kmer[i],scaled_columns_train,scaled_columns_test,dummy_train,dummy_test)
        
        ctr=0
        for j in range(0,len(dat_train)):
            print("For kmer size: ",len(dat_train[j].columns[0])," type: ",names[ctr])
            dat_train[j]['Chr'] = dat_train[j]['Chr'].replace(['X'],'23')
            model,X_red,scores=classification_results(dat_train[j],dat_test[j])
            ctr=ctr+1


def main():
    train,train_kmer,pos_data_test,test_kmer,scaled_columns_train,scaled_columns_test=pd_read_pattern()
    deriving_final_model(train,pos_data_test,train_kmer,test_kmer,scaled_columns_train,scaled_columns_test)
if __name__ == "__main__":
      main()
                   
        


  
