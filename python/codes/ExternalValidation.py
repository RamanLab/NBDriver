#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shayantan Banerjee
path = "/data/shayantan/NBDriver/python/data"
This program performs independent validation using NBDriver
"""

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
from sklearn import metrics
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix


def read_pretrained_model_params():
    path = "/data/shayantan/NBDriver/python/data"
    model = pd.read_pickle(path+"/NBDriver_saved_models_params/final_model.sav")
    tf_vocab=pd.read_pickle(path+"/NBDriver_saved_models_params/TFIDF_vect.sav")
    top_feats=pd.read_pickle(path+"/NBDriver_saved_models_params/top_feats.sav")
    scaler=pd.read_pickle(path+"/NBDriver_saved_models_params/scaler.sav")
    return model,tf_vocab,top_feats,scaler

def read_train_and_test_data():
    path = "/data/shayantan/NBDriver/python/data"

    #Martelotto et al. data for a neighborhood size of 10
    w10k_marte=pd.read_csv(path+"/marte_l_window_10kmer.txt",sep="\t")
    martelotto=pd.read_csv(path+"/Martelotto_test_final.csv")
    #CMC data for a neighborhood size of 10
    w10k_cmc=pd.read_csv(path+'/cmc_window_10kmer.txt',sep="\t")
    cmc=pd.read_csv(path+'/CMC_test_final.csv')    
    #Rheinbay et al. data for a neighborhood size of 10
    w10k_rb=pd.read_csv(path+'/rb_window_10kmer.txt',sep="\t")
    rb=pd.read_csv(path+'/RB_final_test.csv')
    #CVOM data for a neighborhood size of 10
    w10k_cvom=pd.read_csv(path+'/cvom_window_10kmer.txt',sep="\t")
    cvom=pd.read_csv(path+'/CVOM_final_test.csv')
    #OVGBM data for a neighborhood size of 10
    w10k_ovgbm=pd.read_csv(path+'/ov_gbm_window_10kmer.txt',sep="\t")
    ovgbm=pd.read_csv(path+'/OVGBM_final_test.csv')
    return w10k_cmc,w10k_cvom,w10k_marte,w10k_ovgbm,w10k_rb,cmc,martelotto,rb,cvom,ovgbm




class KDEClassifier(BaseEstimator, ClassifierMixin):
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


def annoate_mutations_using_nbdriver(vect,nbd_test,sc,Type_test,Label_test,Chr_test,scaled_feats_test):
    #scaled_feats_test_1=scaled_feats_test.drop(['Gene.name','ID'],axis=1)
    count_vector_test=vect.transform(preprocess(nbd_test,4))
    df_test=pd.DataFrame(count_vector_test.todense(),columns=vect['cv1'].get_feature_names())
    df_test=pd.DataFrame(sc.transform(df_test),columns=df_test.columns)
    df_test['Type']=Type_test
    df_test['Label']=Label_test
    df_test['Chr']=Chr_test
    df_comb_test=pd.concat([df_test, scaled_feats_test], axis=1)
    df_comb_test = df_comb_test.loc[:,~df_comb_test.columns.duplicated()]
    df_comb_test,to_delet=clean_dataset(df_comb_test)
    return df_comb_test
def preprocess(data,k):
    copy = [[x[i:i+k] for i in range(len(x)-k+1)] for x in data]
    copy=[" ".join(review) for review in copy]
    return copy
def run_nbdriver(df_test,scaled_feats_test,vect,sc,model,cols):
    df_4_tf_test=annoate_mutations_using_nbdriver(vect,df_test['new_nbd'].tolist(),sc,df_test['Type'].tolist(),df_test["Label"].tolist(),df_test['Chromosome'].tolist(),scaled_feats_test)
    preds=print_metrics(model,df_4_tf_test,cols)
    return preds
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_delete = df.isin([np.nan, np.inf, -np.inf]).any(1)
    indices_to_keep=~indices_to_delete
    return df[indices_to_keep].astype(np.float64),df[indices_to_delete].astype(np.float64)

def print_metrics(model,test,cols):
    test_x=test.drop(['Label'],axis=1)
    test_y=test['Label']
    y_probs = model.predict_proba(test_x[cols])[:,1]
    y_test_predictions = np.where(model.predict_proba(test_x[cols])[:,1] > 0.119, 2, 1)
    n_levels=test_y.value_counts().count()
    if(n_levels==1):
        #print(test_x.shape)
        return y_test_predictions
    else:
        mcc=metrics.matthews_corrcoef(test_y,y_test_predictions)
        tn, fp, fn, tp = confusion_matrix(test_y, y_test_predictions).ravel()
        ppv=tp/(tp+fp)
        npv=tn/(tn+fn)
        sen=tp/(tp+fn)
        spe=tn/(tn+fp)
        score=ppv+npv+sen+spe
        #y_test_predictions=model.predict(test_x[cols])
        sensi= sensitivity_score(test_y, y_test_predictions, pos_label=2)
        speci=specificity_score(test_y,y_test_predictions,pos_label=2)
        accu=accuracy_score(test_y,y_test_predictions)
        auro=roc_auc_score(test_y,y_test_predictions)
        #acc=accuracy_score(test_y,y_test_predictions)
        print("Composite Score for Martelotto et al.: ", score)
        return y_test_predictions

def main():

    #read the feature matrices for each independent test set and the pretrained model params
    w10k_cmc,w10k_cvom,w10k_marte,w10k_ovgbm,w10k_rb,cmc,martelotto,rb,cvom,ovgbm=read_train_and_test_data()
    model,tf_vocab,top_feats,scaler=read_pretrained_model_params()
    #Run NBDriver for Martelotto et al.    
    w10k_marte['Chromosome'] = w10k_marte['Chromosome'].replace(['X'],'23')
    p1=run_nbdriver(w10k_marte,martelotto,tf_vocab,scaler,model,top_feats)
    #Run NBDriver for CMC
    w10k_cmc['Chromosome'] = w10k_cmc['Chromosome'].replace(['X'],'23')
    p2=run_nbdriver(w10k_cmc,cmc,tf_vocab,scaler,model,top_feats)
    print("Accuracy (CMC): ",len(p2[p2==2])/len(p2))
    #Run NBDriver for Catalog of Validated Oncogenic Mutations from the Cancer Genome Interpreter
    w10k_cvom['Chromosome'] = w10k_cvom['Chromosome'].replace(['X'],'23')
    p3=run_nbdriver(w10k_cvom,cvom,tf_vocab,scaler,model,top_feats)
    print("Accuracy (CVOM): ",len(p3[p3==2])/len(p3))
    #Run NBDriver for OVGBM
    w10k_ovgbm['Chromosome'] = w10k_ovgbm['Chromosome'].replace(['X'],'23')
    p4=run_nbdriver(w10k_ovgbm,ovgbm,tf_vocab,scaler,model,top_feats)
    print("Accuracy (OVGBM): ",len(p4[p4==2])/len(p4))
    #Run NBDriver for Rheinbay
    w10k_rb['Chromosome'] = w10k_rb['Chromosome'].replace(['X'],'23')
    p5=run_nbdriver(w10k_rb,rb,tf_vocab,scaler,model,top_feats)
    print("Accuracy (Rheinbay): ",len(p5[p5==2])/len(p5))
    
if __name__ == "__main__":
      main()
                   


