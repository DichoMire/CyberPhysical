import pandas as pd, matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

#Function that creates and displays a confusion matrix graph
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
    #If True, we make use of our internal three datasets. Otherwise we predict the original Testing DF.
    workingOnTraining = True
    finalTest = True

    if workingOnTraining :
        #Extract the data as DFs
        trainingDf = pd.read_csv("Training.csv")
        validationDf = pd.read_csv("Validation.csv")

        testingDf = pd.read_csv("Testing.csv")

        #Split into Data and Labels
        trainingData = trainingDf.iloc[:, :-1]
        trainingLabels = trainingDf["Labels"]

        validationData = validationDf.iloc[:, :-1]
        validationLabels = validationDf["Labels"]

        testingData = testingDf.iloc[:, :-1]
        testingLabels = testingDf["Labels"]
    else :
        #Extract the data as DFs
        trainingDf = pd.read_csv("TrainingFull.csv")
        testingDf = pd.read_csv("TestingFull.csv")

        #Split into Data and Labels
        trainingData = trainingDf.iloc[:, :-1]
        trainingLabels = trainingDf["Labels"]

        testingData = testingDf.iloc[:, :]

    """Initialize and fit the model
    
    We will be using a support vector classifier, due to its high accuracy.
    """
    #model = LinearRegression()
    #model = LinearSVR()
    #model = DecisionTreeClassifier()
    model = SVC()

    #Fit the model using the Training Dataset
    model.fit(trainingData, trainingLabels)

    if workingOnTraining :
        if not finalTest :
            #Calulate the predicted values for the validation DF.
            validationPredicted = model.predict(validationData)

            """Required if using a non-classifier model"""
            #validationPredicted = list(map(lambda x : 0 if x <= 0.5 else 1, validationPredicted))

            #Calculate F1 score and confusion matrix for hyperparameter optimization purposes
            f1Res = f1_score(validationLabels, validationPredicted)
            print(f1Res)

            generateConfusionMatrix(validationLabels, validationPredicted)
        else :
            """Once we've tuned the hyperparameter, we can try out our model on the test set.
            We should only use that once, as making decisisons off of more predictions on the test set is pointless.
            """
            testingPredicted = model.predict(testingData)

            f1Res = f1_score(testingLabels, testingPredicted)
            print(f1Res)

            generateConfusionMatrix(testingLabels, testingPredicted)
    else :
        #Predict the values for the actual Test set
        testingPredicted = model.predict(testingData)
        testingData["Labels"] = testingPredicted

        #Export results as .txt
        testingData.to_csv("TestingResults.txt", index=False, header=False)

