from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X,y=load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
nb=GaussianNB()
nb.fit(X_train ,y_train)
predicted_probas=nb.predict_proba(X_test)
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_roc(y_test,predicted_probas)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits as load_data
from sklearn.model_selection import cross_val_predict
X,y=load_digits(return_X_y=True)
classifier=RandomForestClassifier()

predictions=cross_val_predict(classifier,X,y)
plot=skplt.metrics.plot_confusion_matrix(y,predictions,normalize=True)
plt.show()