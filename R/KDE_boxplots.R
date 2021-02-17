library(reshape2)
library(ggplot2)
n=c("w1_cv2","w1_cv3","w1_tf2","w1_tf3","w2_cv2","w2_cv3","w2_cv4","w2_tf2","w2_tf3","w2_tf4",
    "w3_cv2","w3_cv3","w3_cv4","w3_tf2","w3_tf3","w3_tf4","w4_cv2","w4_cv3","w4_cv4","w4_tf2","w4_tf3","w4_tf4",
    "w5_cv2","w5_cv3","w5_cv4","w5_tf2","w5_tf3","w5_tf4",
    "w6_cv2","w6_cv3","w6_cv4","w6_tf2","w6_tf3","w6_tf4",
    "w7_cv2","w7_cv3","w7_cv4","w7_tf2","w7_tf3","w7_tf4",
    "w8_cv2","w8_cv3","w8_cv4","w8_tf2","w8_tf3","w8_tf4",
    "w9_cv2","w9_cv3","w9_cv4","w9_tf2","w9_tf3","w9_tf4",
    "w10_cv2","w10_cv3","w10_cv4","w10_tf2","w10_tf3","w10_tf4")

w10_orig=cbind(KDE_KMER_orig[,c(grep("w10_",colnames(KDE_KMER_orig)))],w10_ohe=KDE_OHE_orig[,c("w10")])
w10_rand=cbind(KDE_KMER_rand[,c(grep("w10_",colnames(KDE_KMER_rand)))],w10_ohe=KDE_OHE_rand[,c("w10")])

orig=data.frame(w1_orig$w1_tf2,w2_orig$w2_ohe,w3_orig$w3_cv2,w4_orig$w4_tf3,w5_orig$w5_cv3,w6_orig$w6_tf4,w7_orig$w7_cv2,w8_orig$w8_tf3,w9_orig$w9_tf3,w10_orig$w10_tf4)
rand=data.frame(w1_rand$w1_cv3,w2_rand$w2_cv2,w3_rand$w3_cv2,w4_rand$w4_cv4,w5_rand$w5_tf2,w6_rand$w6_cv2,w7_rand$w7_cv4,w8_rand$w8_cv2,w9_rand$w9_cv2,w10_rand$w10_cv2)
for(i in (1:10)){
  p=(sum(rand[,i] >= median(orig[,i]))+1)/(50 + 1)   
  print(p)
}
#download "Original_KDE.txt" and "Random_KDE.txt" from /data/shayantan/NBDriver/data and store them as orig_b and rand_b respectively
orig_b=data.frame(Window_Size=as.character(unlist(lapply(1:10, function(x) rep(x,30)))), JS_distance=melt(orig)$value, Type=rep("Original",300))
rand_b=data.frame(Window_Size=as.character(unlist(lapply(1:10, function(x) rep(x,30)))), JS_distance=melt(rand)$value, Type=rep("Random",300))
df=rbind(orig_b,rand_b)
df$Window_Size=factor(df$Window_Size)
levels(df$Window_Size)=c("1","2","3","4","5","6","7","8","9","10")
p2 <- ggplot(df, aes(x=Window_Size, y=JS_distance, fill=Type)) + 
  geom_boxplot() +
  facet_wrap(~Window_Size, scale="free")
p2
