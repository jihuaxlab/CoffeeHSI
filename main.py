import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,mean_absolute_error
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import seaborn
seaborn.set(style='whitegrid',font_scale=1.5)

np.random.seed(42)
REGRESSOR = 'LGB'

x = np.loadtxt('x.txt')
labels = ['F1','F2','F3','F4','F5','F6','F7','F8','P1','P2','P3','P4','P5']

def kfo(x,y,name):
    kf = KFold(n_splits=2,shuffle=True)
    if name == 'Qgrade':
        pred = np.zeros(len(y))
        predr = np.zeros(len(y))
    else:
        pred = np.zeros(len(y)).astype('str')
    
    for i,(train,test) in enumerate(kf.split(x)):
        xt = x[train]
        yt = y[train]
        xv = x[test]
        yv = y[test]
        if REGRESSOR == 'SVM':
            clf = SVC()
        elif REGRESSOR == 'KNN':
            clf = KNeighborsClassifier()
        elif REGRESSOR == 'RF':
            clf = RandomForestClassifier(max_depth=5,n_estimators=50)
        elif REGRESSOR == 'XGB':
            clf = XGBClassifier(max_depth=5,n_estimators=50)
        else:
            clf = LGBMClassifier(max_depth=5,n_estimators=50,random_state=42,n_jobs=-1)
        
        clf.fit(xt,yt)
        pred[test] = clf.predict(xv)
        if name == 'Qgrade':
            clfr = LGBMRegressor(max_depth=5,n_estimators=50,random_state=42,n_jobs=-1)
            clfr.fit(xt,yt)
            predr[test] = clfr.predict(xv)

    if name == 'Label': 
        labeltxt = labels
    else:
        labeltxt = list(set(y))
    
    confusion = confusion_matrix(y,pred,labels=labeltxt)
    plt.figure(figsize=(8,6))
    seaborn.heatmap(confusion,annot=True,cbar=False,fmt='d',
            xticklabels=labeltxt,
            yticklabels=labeltxt)
    plt.xlabel('Pred')
    plt.ylabel('True')
    acc = accuracy_score(y,pred)
    plt.title('%s, accuracy=%.3f' % (name,acc))
    plt.show()
    if name == 'Qgrade':
        r2 = r2_score(y,predr)
        mae = mean_absolute_error(y,predr)
        plt.figure()
        plt.title('%s, MAE=%.3f, r2=%.3f' % (name,mae,r2))
        seaborn.boxplot(y,predr,color='cyan',fliersize=0)
        plt.show()

score = [79,79,79,78,78,78,77,77,83,82,81,81,80]
y = []
quality = []
qgrade = []
for i in range(1950):
    y.append(labels[i//150])
    quality.append(labels[i//150][0])
    qgrade.append(score[i//150])

y = np.hstack((y,y))
quality = np.hstack((quality,quality))
qgrade = np.hstack((qgrade,qgrade))
kfo(x,y,'Label')
kfo(x,quality,'Quality')
kfo(x,qgrade,'Qgrade')

x = np.loadtxt('place-x.txt')
y = np.loadtxt('place-y.txt',dtype='str')
kfo(x,y,'Place')

continent = []
for i in range(len(y)):
    if y[i] in ['Indonesia','China']:
        continent.append('Asia')
    elif y[i] in ['Ethiopia','Rwanda','Kenya','Uganda']:
        continent.append('Africa')
    else:
        continent.append('America')

continent = np.array(continent)
kfo(x,continent,'Continent')