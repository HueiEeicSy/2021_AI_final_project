import glob

for imageName in glob.glob('*.txt'):
    newList = []
    f = open(imageName)
    for line in f.readlines():
        tempList = line.split()
        newTempList = [tempList[0], tempList[5], tempList[1], tempList[2], tempList[3], tempList[4]]
        newList.append(newTempList)
    f.close
    w = open(imageName, 'w')
    for line in newList:
        newString = line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' ' + line[4] + ' ' + line[5] + '\n'
        w.write(newString)