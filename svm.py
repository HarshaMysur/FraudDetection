from sklearn import svm
import pandas as pd
from sklearn.cross_validation import train_test_split

data = pd.read_csv('creditcard.csv')

x= data.iloc[:,1:30]
y = data.iloc[:,30]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


clf = svm.SVC(gamma=0.001,C=100)

clf.fit(x_train,y_train)

predictions = clf.predict(x_test)

y_test = y_test.values
tp,fp,tn,fn = 0,0,0,0
for i in range(len(predictions)):
    if (predictions[i] == y_test[i]== 1):
        tp += 1
    elif (predictions[i] == y_test[i]== 0):
        tn += 1
    elif (predictions[i] == 0 and y_test[i]== 1):
        fn += 1
    elif (predictions[i] == 1 and y_test[i]== 0):
        fp += 1

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print ("precision:", precision)
print ("recall:", recall)