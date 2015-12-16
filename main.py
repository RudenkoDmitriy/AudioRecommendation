#-*- coding: utf-8 -*-

from database import createDatasetFileDB, createTargetFileDB, openDB
from algorithm import recommendModel, checkModel, testKNNMetric, testKNNNeingh, testSVCKernel, testGBCEst,testGBCLoss, testBattle

from os import listdir

import time

if __name__ == '__main__':
    time.clock()
    count = 10000
    # createDatasetFileDB("datasetdb100000.tit", "dataset", count)
    # createTargetFileDB("targetdb100000.tit", "target", count)
    dbds = openDB("datasetdb10000.tit")
    dbtg = openDB("targetdb10000.tit")
    model = recommendModel(dbds, count)
    accur, result = checkModel(dbtg, model, "target", "trueTarget", "falseTarget", count)
    print "Accurance :", accur
    print "Time : ", time.clock()
    listName = listdir("target")
    for i in range(0, listName.__len__()):
        if result[i] == 1:
             print listName.pop()

    # testKNNNeingh(dbds, count)
    # testKNNMetric(dbds, count)
    # testSVCKernel(dbds, count)
    # testGBCEst(dbds, count)
    # testGBCLoss(dbds, count)
    #testBattle(dbds, count)
