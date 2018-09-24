from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz

#class that represents P(att = attVal | Label = giveCon)
class condProb:
    def __init(self):
        self.att = ""
        self.attVal = ""
        self.givenCon = ""
        self.val = -1

#results of this function are based into naiveBayes
def getConditionalProbs(tdata, fvArray, cLabels):
    condProbs = []
    for feature in fvArray:
        featureName = feature[0]
        #print("feature: " + featureName)
        for fvalue in feature[1]:
            for label in cLabels:
                #print("      val = " + fvalue + " label = " + label)
                prob = condProb()
                prob.att = featureName
                prob.attVal = fvalue
                prob.givenCon = label
                prob.val =  len( tdata.loc[(tdata[featureName] == fvalue) & (tdata["Label"] == label)] ) / len(tdata.loc[tdata["Label"] == label])
                #print("              att = " + str(prob.att) + " attVal = " + str(prob.attVal) + " givenCon = " + str(prob.givenCon) + " val = " + str(prob.val))
                condProbs.append(prob)

    return condProbs


#makes label predictions based on info in conditionProbs
def naiveBayes(trainData, testData, conditionProbs, labels, colNames):
    results = []
    labelProbs = []
    for label in labels:
        labelProbs.append(len(trainData[trainData["Label"] == label])/len(trainData))

    for index, row in testData.iterrows(): #for each row of the test data
        #print( index, row["Is_Home_or_Away"], row["Is_Opponent_in_AP25_Preseason"], row["Media"])
        lblCount = 0
        importantProb = []
        for label in labels: #for each class label
            classConProb = 1
            for colName in colNames: #for each column in the row
                for conprob in conditionalProbs: #search conditionalProbs for the right conditional prob based on column name, column value, and current label
                    if (conprob.att == colName and conprob.attVal == row[colName] and conprob.givenCon == label): #if match is found
                        classConProb = classConProb * conprob.val
                        break
            importantProb.append(labelProbs[lblCount] * classConProb)  
            lblCount = lblCount + 1

        if(importantProb[0] >= importantProb[1]):
            results.append(labels[0])
        else:
            results.append(labels[1])

    return results


traindf = pd.read_csv('train5.csv')
testdf = pd.read_csv('test5.csv')
conditionalProbs = getConditionalProbs(traindf, [ ["Is_Home_or_Away" , ["Home","Away"] ] , ["Is_Opponent_in_AP25_Preseason" , ["In","Out"] ] , ["Media" , ["1-NBC","2-ESPN","3-FOX","4-ABC","5-CBS"] ] ], ["Win","Lose"] )
labels = naiveBayes(traindf, testdf, conditionalProbs, ["Win","Lose"], ["Is_Home_or_Away" , "Is_Opponent_in_AP25_Preseason" , "Media" ])
actualLabels = testdf["Label"].values
print("Bayes Predictions:")
print(labels)

# print("\n")
# print("Actual       Bayes")
# for i in range(len(actualLabels)):
#     print(actualLabels[i] + "       " + labels[i])

#calculate accuracy
print("\n")
correctCount = 0
for i in range(len(actualLabels)):
    if labels[i] == actualLabels[i]:
        correctCount = correctCount + 1
print("Accuracy = " + str(correctCount/len(labels)))

#calculate precision
truePositives = 0
falsePositives = 0
for i in range(len(actualLabels)):
    if labels[i] == 'Win':
        if actualLabels[i] == 'Win':
            truePositives += 1
        else:
            falsePositives += 1

precision = truePositives / (truePositives + falsePositives)
print("precision = " + str(precision))

#calculate recall
falseNegatives = 0
for i in range(len(actualLabels)):
    if labels[i] == 'Lose':
        if actualLabels[i] == 'Win':
            falseNegatives += 1

recall = truePositives / (truePositives + falseNegatives)
print("recall = " + str(recall))

#F1 score
if (precision + recall) != 0:
    f1 =  2*((precision*recall)/(precision+recall))
else:
    f1 = 0
print("F1 = " + str(f1))