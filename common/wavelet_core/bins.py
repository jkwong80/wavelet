
class waveletBin:

    def __init__(self, parentList, phase):
        self.children = []
        self.chan = []

        if phase == -1:   # Head Node
            for i in range(parentList):
                self.chan.append([i])
        else:
            for i in range(phase,len(parentList),2):
                if i < len(parentList)-1:
                    self.chan.append([val for sublist in parentList[i:i+2] for val in sublist])

        if len(self.chan) > 2:
            self.children.append(waveletBin(self.chan,0))
            self.children.append(waveletBin(self.chan,1))
        elif len(self.chan) == 2:
            self.children.append(waveletBin(self.chan,0))


class waveletBinTree:

    def __init__(self, numBins):
        self.waveBins = waveletBin(numBins,-1)


    def printTreeBins(self):
        self.printTreeKernel(self.waveBins)

    def printTreeKernel(self, inBin):
        print inBin.chan

        for i in range(len(inBin.children)):
            self.printTreeKernel(inBin.children[i])

    def flatList(self,level=None):
        FL = self.flatListKernel(self.waveBins, [])
        FL = [val for sublist in FL for val in sublist]
        tupList = []
        for item in FL:
            if level is None:
                tupList.append((item, (len(item)-1)*10000 + item[0]))
            else:
                if len(item) == 2**level:
                    tupList.append((item, (len(item) - 1) * 10000 + item[0]))

        tupList.sort(key=lambda tup: tup[1])
        return [x[0] for x in tupList ]


    def flatListKernel(self, inBin, inFL):

        inFL.append(inBin.chan)
        for i in range(len(inBin.children)):
            inFL = self.flatListKernel(inBin.children[i],inFL)

        return inFL