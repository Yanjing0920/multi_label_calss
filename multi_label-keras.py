import scipy
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


data, meta = scipy.io.arff.loadarff('data1.arff')
df = pd.DataFrame(data)
#print(df)

X = df.iloc[:,0:43].values
y = df.iloc[:,43:47].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(y_train)
print(y_test)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)
#(y_train)
#print(y_test)

def deep_model(feature_dim,label_dim):
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(50, activation='relu', input_dim=feature_dim))
    model.add(Dense(110, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_deep(X_train,y_train,X_test,y_test):
    feature_dim = X_train.shape[1]
    label_dim = y_train.shape[1]
    model = deep_model(feature_dim,label_dim)
    model.summary()
    History = model.fit(X_train,y_train,batch_size=6, epochs=30,verbose=2,validation_data=(X_test,y_test))
    E = History.epoch.__len__()
    #print(E)

    prediction=model.predict(X_test)
    print(prediction)

    [rows, cols] = prediction.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if prediction[i, j] < 0.2:
                prediction[i, j] = 0
            else:
                prediction[i, j] = 1
    print(prediction)

    #plt.title('Result')
    plt.plot(np.arange(0, E), History.history["loss"], color='green', label='train_loss')
    plt.plot(np.arange(0, E), History.history["val_loss"], color='red', label='val_loss')
    plt.plot(np.arange(0, E), History.history["acc"], color='skyblue', label='train_acc')
    plt.plot(np.arange(0, E), History.history["val_acc"], color='blue', label='val_acc')

    plt.legend()  #显示图例

    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.show()
    return train_deep

train_deep(X_train,y_train,X_test,y_test)