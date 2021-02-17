#author: Shayantan
#Plot the results from the repeated CV experiments using boxplots. Total 21 feature-classifier pairs
library(stringr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(readr)
library(gtools)
final_df=matrix(,nrow=30,ncol=0)
files_kmer=c("KMER_RF_new","KMER_ET_new","KMER_KDE_new")
files_pos=c("POS_RF_new","POS_ET_new","POS_KDE_new")
str_names=c("brf","et","kde")
names=c()
for(k in 1:3){
  print(k)
  print(dim(final_df))
  nm <- list.files(path=paste("/data/shayantan/NBDriver/data/",files_kmer[k],"/",sep=""))
  for(i in 1:40){
    item=nm[i]
    filename=paste(paste("/data/shayantan/NBDriver/data/",files_kmer[k],"/",sep=""),item,sep="")
    df <- read.csv(filename)
    if(parse_number(item) == 1){
      colnames(df)=c("df_2_cv","df_3_cv","df_2_tf","df_3_tf")
    }
    else{
      colnames(df)=c("df_2_cv","df_3_cv","df_4_cv","df_2_tf","df_3_tf","df_4_tf")
    }
    Median=as.data.frame(apply(df,2,median))
    
    final_df=cbind(final_df,df[,which.max(Median[,1])])
    #print(dim(final_df))
    df_name=paste(str_extract(item, "window_[0-9]+_[a-z]{1,3}"),"_",colnames(df)[which.max(Median[,1])],"_",str_names[k],sep="")
    names=c(names,df_name)
      }
  nm_pos <- list.files(path=paste("/data/shayantan/NBDriver/data/",files_pos[k],"/",sep=""))
  for(j in 1:4){
    item=nm_pos[j]
    filename=paste(paste("/data/shayantan/NBDriver/data/",files_pos[k],"/",sep=""),item,sep="")
    df_pos <- read.csv(filename)
    names_pos=paste(colnames(df_pos),"_",strsplit(item,".csv")[[1]][1],sep="")
    df_pos=as.matrix(df_pos)
    names=c(names,names_pos)
    #print(dim(final_df))
    final_df=cbind(final_df,df_pos)
  }
  print(dim(final_df))
  
 }
final_df=as.data.frame(final_df)
colnames(final_df)=names

#best_AUC
auc=final_df[, grep(pattern="auc", colnames(final_df))]
auc_med=apply(auc,2,median)
print(auc_med[which.max(apply(auc,2,median))])
#best_Sensitivity
sen=final_df[, grep(pattern="sen", colnames(final_df))]
sen_med=apply(sen,2,median)
print(sen_med[which.max(apply(sen,2,median))])
#best_Specificity
spe=final_df[, grep(pattern="spe", colnames(final_df))]
spe_med=apply(spe,2,median)
print(spe_med[which.max(apply(spe,2,median))])
#best_MCC
mcc=final_df[, grep(pattern="mcc", colnames(final_df))]
mcc_med=apply(mcc,2,median)
print(mcc_med[which.max(apply(mcc,2,median))])
#window wise best results
kmer_matches=paste("^window_",as.character(1:10),"_",rep(c("auc","spe","sen","mcc"),10),sep="")
pos_matches=paste("^w",as.character(1:10),"_",rep(c("auc","spe","sen","mcc"),10),sep="")
for(i in 1:40){
  w=final_df[,c(grep(paste(kmer_matches[i],"|",pos_matches[i],sep=""),names))]
  print(colnames(w[which.max(apply(w,2,median))]))
  print(median(w[,which.max(apply(w,2,median))]))
}
#Take each metric (Sen,spe,auc,mcc) one at a time and execute the following codes
sen=matrix(,30,10)
spe=matrix(,30,10)
auc=matrix(,30,10)
mcc=matrix(,30,10)
keys=c("sen","spe","auc","mcc")
sen_names=c()
spe_names=c()
auc_names=c()
mcc_names=c()
#In the code below this time we are running sen only. 
for(i in 1:10)
  { kmer_matches=paste("window_",as.character(i),"_","sen",sep="") #use  kmer_matches=paste("window_",as.character(i),"_","spe",sep="") for specificity and so on
    pos_matches=paste("w",as.character(i),"_","sen",sep="") #    pos_matches=paste("w",as.character(i),"_","spe",sep="") for specificity and so on
    print(pos_matches)
    med=data.frame(val=apply(final_df[,c(names[grep(pos_matches,names)],names[grep(kmer_matches,names)])],2,median))
    #print(head(med,10))
    sen[,i]=final_df[,rownames(med)[which.max(med[,1])]] # use spe[,i]=final_df[,rownames(med)[which.max(med[,1])]] for specificity and so on
    sen_names[i]=rownames(med)[which.max(med[,1])] # use     spe_names[i]=rownames(med)[which.max(med[,1])] for specificity and so on

}
sen=data.frame(sen) #use spe =data.frame(spe) for specificity and so on
colnames(sen)=sen_names # use colnames(spe)=spe_names for specificity and so on
print(sen) # use print(spe) for specificity and so on


#plot boxplots
plot_boxplots<-function(x,name_str,n){
  #x is the dataframe containing the best metrics for every window size 
  #n is the number of repeated cv experiments
x_melt=melt(x)
x_melt$window_size=c(rep(1,n),rep(2,n),rep(3,n),rep(4,n),rep(5,n),rep(6,n),rep(7,n),rep(8,n),rep(9,n),rep(10,n))
x_melt[grep("pos",as.character(x_melt$variable)),"features"]="OHE"
x_melt[grep("tf",as.character(x_melt$variable)),"features"]="TF"
x_melt[grep("cv",as.character(x_melt$variable)),"features"]="CV"
colnames(x_melt)=c("variable",name_str,"Window_Size","Features")
x_melt$Window_Size=factor(x_melt$Window_Size)
p=ggplot(x_melt,aes(x=Window_Size,y=x_melt[,2],fill=Features))+
  geom_boxplot()+
  theme(axis.text.x = element_text(face="bold",angle = 45, hjust = 1,size=16),axis.text.y = element_text(face="bold", hjust = 1,size=16),axis.title.x = element_text(size=20, face="bold"),
        axis.title.y = element_text(size=20, face="bold"))+xlab("Window_Size") + ylab(name_str)+theme(legend.title = element_text(color = "black", size = 14),
legend.text = element_text(color = "black", size = 14)
)
plot(p)
return(p) 
}


#Total 10 window sizes and thus 45 pairs. Each window size has the best results for the given metric 
pair_wise_median_test=function(x,metric_name){
  print(toupper(metric_name))
  pairs=combinations(10,2,colnames(x))
  colnames(pairs)=c("pair1","pair2")
  pairs=as.data.frame(pairs)
  df=matrix(,45,3)
  for(i in 1:45){
    m=wilcox.test(x[,as.character(pairs$pair1[i])],x[,as.character(pairs$pair2[i])],paired = TRUE)
    if(m$p.value<0.05){
      df[i,1]=as.character(pairs[i,1])
      df[i,2]=as.character(pairs[i,2])
      df[i,3]=m$p.value
      
    }
    
  }
  df=data.frame(df)
  colnames(df)=c("pair1","pair2","WRS_test_pval")
  print(df[complete.cases(df),])
  return(df[complete.cases(df),])
}
#for sensitivity 
pw_sen=pair_wise_median_test(sen,"sensitivity") #the sen variable must have been calculated by now
melt_sen=plot_boxplots(sen,"Sensitivity",30)

#for specificity
pw_spe=pair_wise_median_test(spe,"specificity") #the spe variable must have been calculated by now
melt_spe=plot_boxplots(spe,"Specificity",30)

#for auc
pw_auc=pair_wise_median_test(auc,"auc") #the auc variable must have been calculated by now
melt_auc=plot_boxplots(auc,"AUC",30)

#for mcc
pw_mcc=pair_wise_median_test(mcc,"mcc") #the mcc variable must have been calculated by now
melt_mcc=plot_boxplots(mcc,"MCC",30)
