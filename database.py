__author__ = 'Dimon'
import wave
import struct
import pickle
import numpy as np
from os import listdir
from scipy import stats

def createStatData((data, nFFT, nchannels)):
    y = np.array(struct.unpack("%dh" % nFFT * nchannels, data))
    y_L = y[::2]
    y_R = y[1::2]
    Y_L = np.fft.fft(y_L, nFFT)
    Y_R = np.fft.fft(y_R, nFFT)
    Y = abs(np.hstack((Y_L[-nFFT/2:-1], Y_R[:nFFT/2])))
    #return {"mean":np.mean(Y), "median":np.median(Y), "stdev":np.std(Y), "skew":stats.skew(Y), "kurt":stats.kurtosis(Y)}
    return [np.mean(Y), np.median(Y), np.std(Y), stats.skew(Y), stats.kurtosis(Y)]

def createMusicData(dirName, count):
    stream = wave.open(dirName, "rb")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = stream.getparams()
    nFFT = ((nframes / count) / 2) * 2
    targetDB = []
    # pool = multiprocessing.Pool()
    # for _ in range(multiprocessing.cpu_count()):
    #     data = [(stream.readframes(nFFT), nFFT, nchannels,) for _ in range(count / multiprocessing.cpu_count())]
    #     for k in pool.map(createStatData, data):
    #         targetDB.extend(createStatData((stream.readframes(nFFT), nFFT, nchannels)))
    while (count != 0):
        targetDB.extend(createStatData((stream.readframes(nFFT), nFFT, nchannels)))
        count -= 1
    stream.close()
    return targetDB

def createTargetDB(dirName, count):
    listOfMusic = listdir(dirName)
    db = np.ndarray((listOfMusic.__len__(), count * 5))
    cn = 0
    for i in range(0,listOfMusic.__len__()):
        db[i][0: count * 5] = createMusicData(dirName + '/' + listOfMusic.pop(), count)
        cn += 1
        print cn
    return db

def createDatasetDB(dirName, count):
    listOfMusicTrue = listdir(dirName + '/true')
    listOfMusicFalse = listdir(dirName + '/false')
    dbTrue = np.ndarray((listOfMusicTrue.__len__(), count * 5 + 1))
    dbFalse = np.ndarray((listOfMusicFalse.__len__(), count * 5 + 1))
    cn = 0
    for i in range(0, listOfMusicTrue.__len__()):
        temp = np.ndarray(count * 5 + 1)
        temp[0:count * 5] = createMusicData(dirName + '/true' + '/' + listOfMusicTrue.pop(), count)[0 : count *5]
        temp[count * 5] = 1
        dbTrue[i][0 : count * 5 + 1] = temp[0 : count * 5 + 1]
        cn+= 1
        print cn
    for i in range(0, listOfMusicFalse.__len__()):
        temp = np.ndarray(count * 5 + 1)
        temp[0:count * 5] = createMusicData(dirName + '/false' + '/' + listOfMusicFalse.pop(), count)[0 : count * 5]
        temp[count * 5] = 0
        dbFalse[i][0 : count * 5 + 1] = temp[0 : count * 5 + 1]
        cn += 1
        print cn
    return np.vstack((dbTrue, dbFalse))

def createDatasetFileDB(nameOfFile, nameOfDir, count):
        with open(nameOfFile, "wb") as f:
            pickle.dump(createDatasetDB(nameOfDir, count), f)

def createTargetFileDB(nameOfFile, nameOfDir, count):
        with open(nameOfFile, "wb") as f:
            pickle.dump(createTargetDB(nameOfDir, count), f)

def openDB(nameOfFileDB):
    with open(nameOfFileDB, "rb") as f:
            return pickle.load(f)