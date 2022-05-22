import pandas as pd, random, matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

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

def splitTrainingSet(trainingDf = None) :
    indexes = trainingDf.index.tolist()

    randTrainingSample = random.sample(indexes, int(len(indexes)/10)*8)
    remaining = [ind for ind in indexes if ind not in randTrainingSample]
    randTestingSample = random.sample(remaining, int(len(remaining)/2))
    randValidationSample = [ind for ind in remaining if ind not in randTestingSample]

    randTrainingSample = trainingDf.iloc[randTrainingSample]
    randTestingSample = trainingDf.iloc[randTestingSample]
    randValidationSample = trainingDf.iloc[randValidationSample]
    testingOriginal = randTestingSample.copy()
    validationOriginal = randValidationSample.copy()

    randValidationSample = randValidationSample.drop(['Labels'], axis = 1)
    randTestingSample = randTestingSample.drop(['Labels'], axis = 1)

    return(randTrainingSample, randTestingSample, randValidationSample, testingOriginal["Labels"], validationOriginal["Labels"])

def generateConfusionMatrix(validationOriginal = None, validationPredicted = None) :
    """The following code is take from:
    https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/

    It uses matplotlib to generate a confusion matrix. This is to make sure that the algorithm performs
    well at prediction both 0s and 1s. 
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=validationOriginal, y_pred=validationPredicted)

    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

if __name__ == '__main__':
    workingOnTraining = True

    #Extract the data as DFs
    trainingDf = fileToDf("TrainingData.txt", True)
    testingDf = fileToDf("TestingData.txt", False)

    #Split training DataFrame into a training and validation DF
    if workingOnTraining :
        (trainingData, testingData, validationData, testingOriginal, validationOriginal) = splitTrainingSet(trainingDf)
    else :
        trainingData = trainingDf
        testingData = testingDf

    """Initialize and fit the model
    
    We will be using a support vector classifier, due to its high accuracy.
    """
    #model = LinearRegression()
    #model = LinearSVR()
    #model = DecisionTreeClassifier()
    model = SVC()


    model.fit(trainingData[range(0,24)], trainingData["Labels"])

    if workingOnTraining :
        validationPredicted = model.predict(validationData[range(0,24)])

        """Required if using a non-classifier model"""
        #validationPredicted = list(map(lambda x : 0 if x <= 0.5 else 1, testingPredicted))

        """Calculate F1 score and confusion matrix for hyperparameter  purposes"""
        f1Res = f1_score(validationOriginal, validationPredicted)
        #print(f1Res)

        generateConfusionMatrix(validationOriginal, validationPredicted)


        """Once we've tuned the hyperparameter, we can try out our model on the test set.
        We should only use that once, as making decisisons off of more predictions on the test set is pointless.
        """
        #testingPredicted = model.predict(testingData[range(0,24)])

        #f1Res = f1_score(testingOriginal, testingPredicted)
        #print(f1Res)

        #generateConfusionMatrix(testingOriginal, testingPredicted)
    else :
        testingPredicted = model.predict(testingData)
        print(testingPredicted)