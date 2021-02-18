#majority voting
#author: Shayantan Banerjee
library(pheatmap)
library(caret)
library(mltools)
martelotto_labels_by_15_tools=read.table("/data/shayantan/NBDriver/R/data/Martelotto_labels.txt",sep = "\t",header = TRUE)
Martelotto_for_validation=read.table("/data/shayantan/NBDriver/R/data/Martelotto_for_validation.txt",sep="\t",header = TRUE)
print("Select tool names from the following list")
print(colnames(martelotto_labels_by_15_tools))
ensemble_of_tools=function(martelotto_labels_by_15_tools,Martelotto_for_validation){
  #selecting ensemble_of_tools(martelotto_labels_by_15_tools,Martelotto_for_validation)
  comb=subset(martelotto_labels_by_15_tools,select=c("NBDriver_int","Mutation_Taster","Condel"))
  preds=apply(comb, 1, Mode)
  Martelotto_for_validation$Funnctional.impact=factor(Martelotto_for_validation$Funnctional.impact)
  c=confusionMatrix(factor(preds),Martelotto_for_validation$Funnctional.impact,positive = "2")
  sen=c$byClass["Sensitivity"][[1]]
  spe=c$byClass["Specificity"][[1]]
  ppv=c$byClass["Pos Pred Value"][[1]]
  npv=c$byClass["Neg Pred Value"][[1]]
  mcc_pred=mcc(factor(Martelotto_labels$NBDriver_int),Martelotto_for_validation$Funnctional.impact)
  print(mcc_pred)
  score=sen+spe+ppv+npv
  print(c)
  print(score)
}
#Output for an ensemble containing NBDriver_int, Mutation Taster, Condel
ensemble_of_tools(martelotto_labels_by_15_tools,Martelotto_for_validation)
