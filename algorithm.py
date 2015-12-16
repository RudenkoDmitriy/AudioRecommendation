__author__ = 'Dimon'
from os import listdir
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def recommendModel(db, count):
    # model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    model = GradientBoostingClassifier(loss='deviance', n_estimators=8, learning_rate=1, max_depth=11, min_samples_split=3, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=10000)
    # model = SVC(C = 1, kernel='rbf')
    # model = LogisticRegression()
    model.fit(db[:,0:count * 5], db[:, count * 5])
    return model

def ROCBattle(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    model =[KNeighborsClassifier(n_neighbors=3, metric = 'euclidean'), LogisticRegression(), SVC(), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)]
    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.7)
    plt.clf()
    plt.figure(figsize=(8,6))
    for model in model:
        model.probability = True
        probas = model.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
        fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
        roc_auc  = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')
    plt.show()

def testKNNMetric(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    testData = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    kfold = 5
    itog_val = {}
    for i in testData:
        scores = cross_validation.cross_val_score(KNeighborsClassifier(metric = i, n_neighbors = 3), train, target, cv = kfold)
        itog_val[i] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False)
    plt.show()

def testGBCLoss(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    testDataLoss = ['deviance', 'exponential']
    kfold = 5
    itog_val = {}
    for i in testDataLoss:
        scores = cross_validation.cross_val_score(GradientBoostingClassifier(loss=i, n_estimators=8, learning_rate=1, max_depth=3, min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=3200), train, target, cv = kfold)
        itog_val[i] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False)
    plt.show()

def testGBCEst(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    testDataEst = [i for i in range(0, 20, 1)][1:]
    kfold = 5
    itog_val = {}
    for i in testDataEst:
        scores = cross_validation.cross_val_score(GradientBoostingClassifier(loss='deviance', n_estimators=i, learning_rate=1, max_depth=3, min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=3200), train, target, cv = kfold)
        itog_val[i] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False)
    plt.show()

def testKNNNeingh(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    testData = [i for i in range(1, 21, 2)]
    kfold = 5
    itog_val = {}
    for i in testData:
        scores = cross_validation.cross_val_score(KNeighborsClassifier(n_neighbors = i), train, target, cv = kfold)
        itog_val[i.__str__()] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False)
    plt.show()

def testSVCKernel(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    def drange(start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step
    testDataKernel =['linear', 'poly', 'rbf', 'sigmoid']
    kfold = 5
    itog_val = {}
    for i in testDataKernel:
        scores = cross_validation.cross_val_score(SVC(kernel=i, degree=100), train, target, cv = kfold)
        itog_val[i.__str__()] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False, fontsize=20)
    plt.show()

def testBattle(db, count):
    train = db[:,0:count * 5]
    target = db[:, count * 5]
    kfold = 5
    itog_val = {}
    models10000 = [("SVC",SVC(C = 1, kernel='rbf')), ('KNN',KNeighborsClassifier(metric = 'euclidean', n_neighbors=7)), ('LogReg',LogisticRegression()), ('GBC',GradientBoostingClassifier(loss='deviance', n_estimators=8, learning_rate=1, max_depth=8, min_samples_split=3, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=10000))]
    models1000 = [("SVC",SVC(C = 1, kernel='rbf')), ('KNN',KNeighborsClassifier(metric = 'manhattan', n_neighbors=11)), ('LogReg',LogisticRegression()), ('GBC',GradientBoostingClassifier(loss='exponential', n_estimators=8, learning_rate=1, max_depth=8, min_samples_split=3, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=10000))]
    models100 = [("SVC",SVC(C = 1, kernel='rbf')), ('KNN',KNeighborsClassifier(metric = 'euclidean', n_neighbors=11)), ('LogReg',LogisticRegression()), ('GBC',GradientBoostingClassifier(loss='exponential', n_estimators=8, learning_rate=1, max_depth=8, min_samples_split=3, min_samples_leaf=2, min_weight_fraction_leaf=0, subsample= 1, max_features='auto', random_state=10000))]
    for (i, j) in models10000:
        scores = cross_validation.cross_val_score(j, train, target, cv = kfold)
        itog_val[i] = scores.mean()
    DataFrame.from_dict(data = itog_val, orient='index').plot(kind='barh', legend = False)
    plt.show()

def checkModel(dbtg, model, nameOfTargetDir, nameOfTrueTargetDir, nameOfFalseTargetDir, count):
        test =[]
        for x in dbtg:
            test.append(model.predict(x))
        count = 0
        trueTargetList = listdir(nameOfTrueTargetDir)
        falseTargetList = listdir(nameOfFalseTargetDir)
        listName = listdir(nameOfTargetDir)
        listNumber = listName.__len__()
        for i in range(0, listNumber):
            if test[i] == 1:
                if listName.pop() in trueTargetList:
                    count += 1
            else:
                if listName.pop() in falseTargetList:
                    count += 1
        return round((float(count)/listNumber) * 100, 5), test

