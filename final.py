# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:57:38 2020

@author: 90544
"""
import pandas as pd
from sklearn import model_selection
#metrikler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
#basit modeller
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#ensemble  modeller
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier


# Veri setinin yüklenmesi
iris_dataset = pd.read_csv('iris.csv')

# Bağımlı ve bağımsız değişkenlerin oluşturulması
X = iris_dataset.values[:, 0:4]
Y = iris_dataset.values[:, 4]

# Veri kümesinin eğitim ve test verileri olarak ayrılması
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

#BASİT MODELLER

# 1. DSupport Vector Classification 

kernel='rbf'
svc = SVC(kernel='rbf',random_state=7)
svc.fit(X_train, Y_train)
predictions = svc.predict(X_test)
print("\n Confusion Matrix(kernel=rbf): \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri(kernel=rbf): \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'))

#kernel='linear' 
svc = SVC(kernel='linear',random_state=7)
svc.fit(X_train, Y_train)
predictions = svc.predict(X_test)
print("\n Confusion Matrix(kernel=linear): \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri(kernel=linear): \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'))
    
#kernel='poly' 
svc = SVC(kernel='poly',random_state=7)
svc.fit(X_train, Y_train)
predictions = svc.predict(X_test)
print("\n Confusion Matrix(kernel=poly): \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri(kernel=poly): \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'))


    
#kernel='sigmoid'
svc = SVC(kernel='sigmoid',random_state=7)
svc.fit(X_train, Y_train)
predictions = svc.predict(X_test)
print("\n Confusion Matrix(kernel=sigmoid): \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri(kernel=sigmoid): \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'))


  # 2- Decision Tree Classifier

# default değerle
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_test)
print(" Decision Tree Classifier")
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")



#paremetreli: (random_state=7,criterion="gini",splitter="best",max_depth=6,min_samples_split=3))
dtc = DecisionTreeClassifier(random_state=7,criterion="gini",splitter="random",max_depth=6,min_samples_split=3)
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_test)
print(" Decision Tree Classifier")
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")



#paremetreli: (random_state=7,criterion="entropy",splitter="best",max_depth=7,min_samples_split=5,min_samples_leaf=3)
dtc = DecisionTreeClassifier(random_state=7,criterion="entropy",splitter="best",max_depth=7,min_samples_split=5,min_samples_leaf=3)
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_test)
print(" Decision Tree Classifier")
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")



   # 3- K-nearest neighbors

#default değerle
print("K-nearest neighbors-default")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


#paremetreli
print("K-nearest neighbors-paremetreli")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='ball_tree', leaf_size=10, p=3, n_jobs=-1)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")




#paremetreli
print("K-nearest neighbors-paremetreli")
knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='brute', leaf_size=40, p=1, n_jobs=-2)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


#ENSEMBLE- KOLLEKTİF MODELLER

# 1. Random Forest Classifier

# #default değerle
print("Random Forest Classifier-default")
rf=RandomForestClassifier()
rf.fit(X_train, Y_train)
predictions = rf.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


# 2. Ada Boost Classifier
#default değerle
print("Ada Boost Classifier-default")
abc=AdaBoostClassifier()
abc.fit(X_train, Y_train)
predictions = abc.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


# AdaBoost-SVC
print("Ada Boost Classifier-SVC algoritması ile")
svc=SVC(probability=True, kernel='linear',random_state=7)
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,algorithm = 'SAMME',learning_rate=1)
abc.fit(X_train, Y_train)
predictions = abc.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")




# 2. Bagging Classifier
#default değerle
print("Bagging Classifier-default")
bc = BaggingClassifier()
bc.fit(X_train, Y_train)
predictions = bc.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


#svc algoritması ile
print("Bagging Classifier-SVC algoritması ile")
svc=SVC(probability=True, kernel='linear',random_state=7)
bc = BaggingClassifier(n_estimators=50,base_estimator=svc)
bc.fit(X_train, Y_train)
predictions = bc.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")


# 3. Voting Classifer
# Voting Classifer: Support Vector Classification -Decision Tree Classifier -K-nearest neighbors 
print("Voting Classifer: Support Vector Classification -Decision Tree Classifier -K-nearest neighbors ")
svc = SVC(kernel='linear',random_state=7)
dtc = DecisionTreeClassifier(random_state=7,criterion="gini",splitter="random",max_depth=6,min_samples_split=3)
knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='ball_tree', leaf_size=10, p=3, n_jobs=-1)
v = VotingClassifier(estimators=[('SVC', svc), ('dtc', dtc), ('knn', knn)], voting='hard')
v = v.fit(X_train, Y_train)
predictions = v.predict(X_test)
print("\n Confusion Matrix: \n",confusion_matrix(Y_test, predictions))
print("\n \n Confusion Matristen Elde Edilen Metrik Değerleri: \n",classification_report(Y_test, predictions))
print('Accuracy degeri:', accuracy_score(Y_test, predictions))
print('Presicion değeri:',precision_score(Y_test, predictions, average = 'macro'))
print('Recall değeri:',recall_score(Y_test, predictions, average = 'macro'))
print('F1-Score:',f1_score(Y_test,predictions,average='micro'))
print('Support değeri:',precision_recall_fscore_support(Y_test, predictions, average = 'macro'),"\n")



