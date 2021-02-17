# -*- coding: utf-8 -*-
"""
@author: Shayantan Banerjee
path = "/data/shayantan/NBDriver/python/data"
This program calculates the original and randomized JS distances between the class-wise estimated densities (using KDE estimation method)
"""

import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy import stats
from scipy import stats
from sklearn.pipeline import Pipeline
from scipy.spatial import distance
from sklearn.utils import shuffle
import scipy as scp
from sklearn.preprocessing import LabelEncoder
import glob


def pd_read_pattern():
    #Loads all the required files into a list
    file_paths_pos = glob.glob(path+"/*pos.txt")
    pos_data=[pd.read_csv(f,sep="\t") for f in file_paths_pos]
    file_paths_kmer = glob.glob(path+"/*kmer.txt")
    kmer_data=[pd.read_csv(f,sep="\t") for f in file_paths_kmer]
    return pos_data,kmer_data

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

def dat_prep(nbd,k,vect_type,Type,Chr,Label):
    
    #Derives the Count Vectorizer or TFIDF scores for a given neighborhood sequence
    """
    Arguments:
        nbd = Column containing the neighborhood sequence
        k=size of kmer
        vect_type= 'CV' for Count Vectorizer or else TFIDF Vectorizer 
        Type=Numerically encoded substitution Type ("A>T" encoded as 1 or "G>C" encoded as 2 and so on)
        Chr= Chromosome number
        Label=Binary label 1=Passenger and 2=Driver
    Returns:
        df= The complete dataframe of TFIDF or CountVect scores plus other features such as chromosome number and substitution type
    """
    if(vect_type=="CV"):
        vect=Pipeline([('cv1',CountVectorizer(lowercase=False))])
    else:
        vect = Pipeline([('cv1',CountVectorizer(lowercase=False)), ('tfidf_transformer',TfidfTransformer(smooth_idf=True,use_idf=True))])

    count_vector=vect.fit_transform(preprocess(nbd,k))
    df=pd.DataFrame(count_vector.todense(),columns=vect['cv1'].get_feature_names())
    df['Type']=Type
    df['Label']=Label
    df['Chr']=Chr
    return df       


def cv_tf_transformation(df):
    #Implements the dat_prep method in a sequential manner
    """
    Arguments:
        df=The initial raw unformatted input for kmer type features
    Returns:
        dat1= returns the data frame 4 possible feature representations for window size 1
        names1= return the names of the 4 possible feature representations for window size 1
        dat2=returns the data frame 6 possible feature representations for window sizes 2-10
        names2= return the names of the 6 possible feature representations for window sizes 2-10

    """
    f=0
    if(len(df['new_nbd'][0])==3):
        df_2_cv=dat_prep(df['new_nbd'].tolist(),2,"CV",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_3_cv=dat_prep(df['new_nbd'].tolist(),3,"CV",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_2_tf=dat_prep(df['new_nbd'].tolist(),2,"tf",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_3_tf=dat_prep(df['new_nbd'].tolist(),3,"tf",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        f=1
        
    else:
        df_2_cv=dat_prep(df['new_nbd'].tolist(),2,"CV",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_3_cv=dat_prep(df['new_nbd'].tolist(),3,"CV",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_4_cv=dat_prep(df['new_nbd'].tolist(),4,"CV",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_2_tf=dat_prep(df['new_nbd'].tolist(),2,"tf",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_3_tf=dat_prep(df['new_nbd'].tolist(),3,"tf",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        df_4_tf=dat_prep(df['new_nbd'].tolist(),4,"tf",df['Type'].tolist(),df['Chromosome'].tolist(),df["Label"].tolist())
        
    if f==1:
        names1=['df_2_cv','df_3_cv','df_2_tf','df_3_tf']
        dat1=[df_2_cv, df_3_cv, df_2_tf,df_3_tf]
        return dat1,names1
    else:
        names2=['df_2_cv','df_3_cv','df_4_cv','df_2_tf','df_3_tf','df_4_tf']
        dat2=[df_2_cv,df_3_cv,df_4_cv, df_2_tf,df_3_tf,df_4_tf]
        return dat2,names2
    
def KDE_estimates(positive,negative,njobs):
    #Implements the KDE estimation procedure separately for the driver (positive) and passenger (negative) data and also calculates the JS distances
    """
    Arguments:
        positive=The feature matrix for driver mutations
        negative=The feature matrix for passenger mutations
        njobs=Number of jobs to run in parallel for the Grid Search
    Returns:
        js_t= Jensen-Shannon distance between the estimated densities
    """
    np.random.seed(333)
    bandwidths = np.logspace(-1, 1, 30)
    grid_pos = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=3,n_jobs=njobs)
    grid_pos.fit(positive)
    kde_pos = KernelDensity(kernel='gaussian', bandwidth = grid_pos.best_params_['bandwidth']).fit(positive)
    grid_neg = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=3,n_jobs=njobs)
    grid_neg.fit(negative)
    kde_neg = KernelDensity(kernel='gaussian', bandwidth = grid_neg.best_params_['bandwidth']).fit(negative)
    score_pos=kde_pos.score_samples(positive)
    score_neg=kde_neg.score_samples(negative)
    js_t=distance.jensenshannon(np.exp(score_pos),np.exp(score_neg))
    return js_t
    
        
def repeated_KDE(pos,neg,n,n_s):
    #repeats the KDE estimation procedure for a predefined number of times
    """
    Arguments:
        pos=The feature matrix for driver mutations
        neg=The feature matrix for passenger mutations
        n=Number of times to run the KDE experiments
        n_s=Number of samples required for the random sampling with replacement process
    Returns:
        js= A list containing the Jensen-Shannon distances
    """
    np.random.seed(333)
    js=[];j_shuff=[];
    for i in range(0,n):
        print(i,end=",")
        d_s=pos.sample(n=n_s, replace=True,random_state=np.random.seed())
        p_s=neg.sample(n=n_s,replace=True,random_state=np.random.seed())
        df=pd.concat([d_s, p_s])  
        
        j=KDE_estimates(d_s.drop('Label',axis=1),p_s.drop('Label',axis=1),100)
        js.append(j)
        
    
    return js

def mean_confidence_interval(data, confidence=0.95):
    #Caluclates the 95% CI for a given list of numbers
    """
    Arguments:
        data=list of numbers
        confidence=desired level of CI, default of 0.95
    Prints:
        The CI's for the given list
         
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.median(a), scp.stats.sem(a)
    h = se * scp.stats.t.ppf((1 + confidence) / 2., n-1)
    print("(",m-h,"," ,m+h,")")
    
def KDE_estimates_kmer():
    #The KDE estimation experiments for kmer based feature representation , 1100 samples were randomly selected by replacement during each run
    """
    Arguments:
        None
    Prints:
        The median JS distances (and CIs) and the randomized JS distances (and CIs) for all window sizes 1-10
        
    """
    pos_data,kmer_data=pd_read_pattern()
    listoflists_k=[];j_r_k=[]
    for i in range(0,len(kmer_data)):
        driver=[];passenger=[]
        dat,names=cv_tf_transformation(kmer_data[i])
        c=0
        for j in dat:
            j_rand_k=[]
            print("Window size: ",len(kmer_data[i]['new_nbd'][0])//2,"\n","Type: ",names[c])
            print("For kmer size: ",len(j.columns[0]))
            j['Chr'] = j['Chr'].replace(['X'],'21')
            driver=j.loc[j['Label']==2]
            passenger=j.loc[j['Label']==1]
            j_kmer=repeated_KDE(driver,passenger,30,1100)
            listoflists_k.append(j_kmer)
            for m in range(0,30):
                print(m,end=",")
                dummy_random_k=j.sample(n=2200,replace=True,random_state=np.random.seed())
                dummy_random_driver_k ,dummy_random_passenger_k = train_test_split(dummy_random_k,test_size=0.5,shuffle=True)
                j_random_k=KDE_estimates(dummy_random_driver_k.drop('Label',axis=1),dummy_random_passenger_k.drop('Label',axis=1))
                j_rand_k.append(j_random_k)
    
            j_r_k.append(j_rand_k)
            c=c+1
            print("Median JS for sample size 2200: ","is ",np.median(j_kmer)," ",mean_confidence_interval(j_kmer),"\n")
            print("Randomized Median JS for sample size 1100: ","is ",np.median(j_rand_k)," ",mean_confidence_interval(j_rand_k),"\n")
            print("Significance: ", scp.stats.mannwhitneyu(j_kmer,j_rand_k))

def KDE_estimates_pos():
    #The KDE estimation experiments for position based feature representation , 1100 samples were randomly selected by replacement during each run
    """
    Arguments:
        None
    Prints:
        The median JS distances (and CIs) and the randomized JS distances (and CIs) for all window sizes 1-10
        
    """
    pos_data,kmer_data=pd_read_pattern()
    new=shuffle_data(pos_data)
    listoflists=[];j_r=[]
    for i in range(0,len(new)):
        driver=[];passenger=[]
        j_pos=[];df=[];j_rand=[]
        print("Window size: ",(len(new[i].columns)-3)//2,"\n")
        non_dummy_cols = ['Type','Label','Chr'] 
        # Takes all 47 other columns
        dummy_cols = list(set(new[i].columns) - set(non_dummy_cols))
        dummy=pd.get_dummies(new[i],columns=dummy_cols)
        dummy['Chr']=new[i]['Chr'];dummy['Type']=new[i]['Type'];dummy['Label']=new[i]['Type']
        dummy['Chr'] = dummy['Chr'].replace(['X'],'21')
        driver=dummy.loc[dummy['Label']==2]
        passenger=dummy.loc[dummy['Label']==1]
        n_s=[1100]
        j_pos=repeated_KDE(driver,passenger,3,1100)
        listoflists.append(j_pos)
        for m in range(0,3):
            print(m,end=",")
            dummy_random=dummy.sample(n=2200,replace=True,random_state=np.random.seed())
            dummy_random_driver ,dummy_random_passenger = train_test_split(dummy_random,test_size=0.5,shuffle=True)
            j_random=KDE_estimates(dummy_random_driver.drop('Label',axis=1),dummy_random_passenger.drop('Label',axis=1))
            j_rand.append(j_random)
            
           
        j_r.append(j_rand)
        print("Median JS for sample size: 1100","is ",np.median(j_pos)," ",mean_confidence_interval(j_pos),"\n")
        print("Randomized Median JS for sample size 1100: ","is ",np.median(j_rand)," ",mean_confidence_interval(j_rand),"\n")
        print("Significance: ", scp.stats.mannwhitneyu(j_pos,j_rand))
        
        
def main():
    print("Calculating JS distances and their statistical significances between the class-wise density estimates (number of iterations=30)")
    KDE_estimates_kmer()
    KDE_estimates_pos()

if __name__ == "__main__":
      main()
