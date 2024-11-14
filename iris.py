from sklearn import datasets
iris=datasets.load_iris()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
print(type(iris))
print(iris.keys())
print(type(iris.data),type(iris.target))
print(iris.data.shape)
print(iris.target_names)

X=iris.data
y=iris.target
print("x=",X)
print(y)

df=pd.DataFrame(X,columns=iris.feature_names)
print(df.head())

from pandas.plotting import scatter_matrix
scatter_matrix(df,alpha=0.2,figsize=(10,10))
#plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)

X_new=np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2]])
'''knn.fit(iris['data'],iris['target'])
prediction=knn.predict(X_new)
print("Predection:{}".format(prediction))'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print("Test set Prediction:\n{}".format(y_pred))
sample_index=0
sample=X_test[sample_index].reshape(1,-1)
prediction=knn.predict(sample)
flower_name=iris.target_names[prediction[0]]
print(f"Predicted flower name for sample {sample_index}: {flower_name}")


print("Accuracy:",knn.score(X_test,y_test))