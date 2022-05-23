import pandas as pd, random

#Convert raw file to pandas DataFrame
def fileToDf(path = None, isTraining = True) :
        #Open and read training file
    f = open(path,"r")
    lines = f.readlines()
    
    #Data structures to parse input
    data = []
    if isTraining :
        labels = []

    for line in lines :
        #Extract the label as int
        input = line.split(",")
        if isTraining :
            label = int(input[-1].strip())
            input.pop(-1)

        #Convert string input array to float input array
        input = map(lambda x : float(x), input)

        data.append(input)
        if isTraining :
            labels.append(label)

    #Create a DataFrame from parsed input
    trainingDf = pd.DataFrame(data=data)
    if isTraining :
        trainingDf["Labels"] = labels

    return trainingDf

#Split Training DataFrame into three DFs:
#Training, Validation and Testing Sets and output them as .csv
def splitTrainingSet(trainingDf = None) :
    indexes = trainingDf.index.tolist()

    randTrainingSample = random.sample(indexes, int(len(indexes)/10)*8)
    remaining = [ind for ind in indexes if ind not in randTrainingSample]
    randTestingSample = random.sample(remaining, int(len(remaining)/2))
    randValidationSample = [ind for ind in remaining if ind not in randTestingSample]

    randTrainingSample = trainingDf.iloc[randTrainingSample]
    randTestingSample = trainingDf.iloc[randTestingSample]
    randValidationSample = trainingDf.iloc[randValidationSample]

    randTrainingSample.to_csv("Training.csv", index=False)
    randValidationSample.to_csv("Validation.csv", index=False)
    randTestingSample.to_csv("Testing.csv", index=False)

if __name__ == '__main__':
    trainingDf = fileToDf("TrainingData.txt", True)
    trainingDf.to_csv("TrainingFull.csv", index=False)

    testingDf = fileToDf("TestingData.txt", False)
    testingDf.to_csv("TestingFull.csv", index=False)

    splitTrainingSet(trainingDf)