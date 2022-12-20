# data
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # for finding the p-value 
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split # to split our data into train and test samples.
#If you want to see how to implement  this split from scratch you can check out my other project Glass Classification using KNN from Scratch in my profile.

from sklearn.metrics import accuracy_score # for calculating our accuracy in the end 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

class OneVsRestSVM:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.clfs = []
        self.y_pred = []
    
    # y를 onehot encoding하는 과정
    def one_vs_rest_labels(self, y_train):
        y_train = pd.get_dummies(y_train)
        return y_train
    
    # encoding된 y를 가져와서 class 개수 만큼의 classifier에 각각 돌리는 과정
    def fit(self, X_train, y_train):
        # y encoding
        y_encoded = self.one_vs_rest_labels(y_train)
        
        for i in range(self.n_classes):
            #clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf = SVM()
            clf.fit(X_train, y_encoded.iloc[:,i])
            self.clfs.append(clf)

    # 각각의 classifier에서 나온 결과를 바탕으로 투표를 진행하는 과정
    def predict(self, X_test):
        vote = np.zeros((len(X_test), 3), dtype=int)
        size = X_test.shape[0]
        
        for i in range(size):
            # 해당 class에 속하는 샘플을 +1 만큼 투표를, 나머지 샘플에 -1 만큼 투표를 진행한다.
            if self.clfs[0].predict(X_test)[i] == 1:
                vote[i][0] += 1
                vote[i][1] -= 1
                vote[i][2] -= 1
            elif self.clfs[1].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] += 1
                vote[i][2] -= 1
            elif self.clfs[2].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] -= 1
                vote[i][2] += 1
    
            # 투표한 값 중 가장 큰 값의 인덱스를 test label에 넣는다
            self.y_pred.append(np.argmax(vote[i]))
           
        #self.y_pred = pd.DataFrame(self.y_pred).replace({0:'setosa', 1:'versicolor', 2:'virginica'})
        return self.y_pred
    
    # accuracy 확인
    def evaluate(self, y_test):
        print('Accuacy : {: .5f}'.format(accuracy_score(y_test, self.y_pred)))

import seaborn as sns
iris =  sns.load_dataset('iris') 
X= iris.iloc[:,:4] #학습할데이터
y = iris.iloc[:,-1] #타겟

#X = X.to_numpy()
#y= y.to_numpy()
# X = pd.read_csv('codes/train_data.csv')
# y = pd.read_csv('codes/label_data.csv')
X = pd.read_csv('./data_for_student\\train\\data.csv')
y = pd.read_csv('./data_for_student\\train\\label.csv')

X = X.transpose()
index = [i for i in range (len(X))]
X.index = index

y=y.transpose()
vector = np.vectorize(np.int_)
y_data = y.index.values.astype(float)
y_data = vector((y_data))
y = pd.Series(y_data)


print(X.shape,'\n',y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)
scaler = StandardScaler() #scaling
X_train = scaler.fit_transform(X_train)#train data로 학습된 Scaler()의 parameter를 통해 test data의 feature 값들이 스케일 되는 것
X_test = scaler.transform(X_test)#train data로부터 학습된 mean값과 variance값을 test data에 적용하기 위해 transform() 메서드를 사용합니다

onevsrest = OneVsRestSVM()
onevsrest.fit(X_train, y_train)
y_pred_rest = onevsrest.predict(X_test)
onevsrest.evaluate(y_test)






























# data = pd.read_csv('D:\\OneDrive\\Documents\\SJTU 과제\\Pattern_recognition\\HW\\datasets\\breast-cancer-wisconsin-data.csv')

# diagnosis_map = {'M':1, 'B':-1}  #We use -1 instead of 0 because of how SVM works. 
# data['diagnosis'] = data['diagnosis'].map(diagnosis_map)

# data.drop(data.columns[[-1, 0]], axis=1, inplace=True) # axis 1 -> columns, 'inplace = True' means we do the drop operation inplace and return None.

# y = data.loc[:, 'diagnosis']  # Select diagnosis column.

# X = data.iloc[:, 1:]  # Select columns other than diagnosis.

# X_normalized = MinMaxScaler().fit_transform(X.values) # Scaling the values in X between (0,1).
# X = pd.DataFrame(X_normalized)

# X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

# clf = SVM()
# clf.init()
# clf.fit(X_train.to_numpy(), y_train.to_numpy())

# y_test_predicted = clf.predict(X_test.to_numpy())

# print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
# print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
# print("precision on test dataset: {}".format(precision_score(y_test.to_numpy(), y_test_predicted)))