import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

data=pd.read_csv('diabetes.csv')
#print(data)
#sliciing 
x=data[['Pregnancies' , 'Glucose' ,'BP' ,'Skin',  'Insulin',   'BMI',  'Pedigree',  'Age']]
#print(x)
#slicing
y=data.Outcome
#print(y.values)

#spliting the data as train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


#appling regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_train)
print(y_pred)

#confusion matrix
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


sns.heatmap(pd.DataFrame(cnf_matrix))
plt.show()
