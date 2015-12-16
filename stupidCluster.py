def createTargetInfo(db):
    maxMean = 0
    maxMedian = 0
    maxStdev = 0
    maxKurt = 0
    maxSkew = 0
    minMean = 9999999
    minMedian = 9999999
    minStdev = 9999999
    minKurt = 9999999
    minSkew = 9999999
    for targetDb in db.values():
        for dirDb in targetDb:
                if (dirDb["mean"] > maxMean):
                    maxMean = dirDb["mean"]
                if (dirDb["median"] > maxMedian):
                    maxMedian = dirDb["median"]
                if (dirDb["stdev"] > maxStdev):
                    maxStdev = dirDb["stdev"]
                if (dirDb["skew"] > maxSkew):
                    maxSkew = dirDb["skew"]
                if (dirDb["kurt"] > maxKurt):
                    maxKurt = dirDb["kurt"]
                if (dirDb["mean"] < minMean):
                    minMean = dirDb["mean"]
                if (dirDb["median"] < minMedian):
                    minMedian = dirDb["median"]
                if (dirDb["stdev"] < minStdev):
                    minStdev = dirDb["stdev"]
                if (dirDb["skew"] < minSkew):
                    minSkew = dirDb["skew"]
                if (dirDb["kurt"] < minKurt):
                    minKurt = dirDb["kurt"]
    return {"maxMean":maxMean, "maxMedian":maxMedian, "maxStdev":maxStdev, "maxKurt":maxKurt, "maxSkew":maxSkew, "minMean":minMean,
            "minMedian":minMedian, "minStdev":minStdev, "minKurt":minKurt, "minSkew":minSkew}

def checkTargetAudio(db, targetInfo):
    result = []
    for targetDb in db.values():
        for dirDb in targetDb:
            if (dirDb["mean"] <= targetInfo["maxMean"]) and (dirDb["median"] <= targetInfo["maxMedian"]) \
                    and (dirDb["stdev"] <= targetInfo["maxStdev"]) and (dirDb["skew"] <= targetInfo["maxSkew"]) \
                    and (dirDb["kurt"] <= targetInfo["maxKurt"]) and ((dirDb["mean"] >= targetInfo["minMean"]) \
                    and (dirDb["median"] >= targetInfo["minMedian"])and (dirDb["stdev"] >= targetInfo["minStdev"])) \
                    and (dirDb["skew"] >= targetInfo["minSkew"]) and (dirDb["kurt"] >= targetInfo["minKurt"]):
                result.append(db.keys().pop())
    return result
