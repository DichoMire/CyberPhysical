import pandas as pd, numpy as np
import rsa

from scipy.optimize import linprog

#Convert testingResults to a Df
def fileToDf(path = None) :
    #Open and read training file
    f = open(path,"r")
    lines = f.readlines()
    
    #Data structures to parse input
    data = []
    labels = []

    for line in lines :
        #Extract the label as int
        input = line.split(",")
        label = int(input[-1].strip())
        input.pop(-1)

        #Convert string input array to float input array
        input = map(lambda x : float(x), input)

        data.append(input)
        labels.append(label)

    #Create a DataFrame from parsed input
    resultsDf = pd.DataFrame(data=data)
    resultsDf["Labels"] = labels

    return resultsDf

if __name__ == '__main__':
    #Code from https://towardsdatascience.com/linear-programming-with-python-db7742b91cb
    pd.set_option('display.max_columns', None)

    input = pd.read_excel("COMP3217CW2Input.xlsx")
    #print(sth)

    user = input["User & Task ID"].map(lambda x: x.split("_")[0][-1])
    task = input["User & Task ID"].map(lambda x: x.split("_")[1][-1])
    task = task.replace("0", "10")

    input["User"] = user
    input["Task"] = task

    input = input.drop("User & Task ID", axis=1)
    input = input.rename(columns = {"Ready Time" : "ReadyT","Maximum scheduled energy per hour" : "MaxH", "Energy Demand" : "EnergyD"})

    variables = []

    restrictions = []

    for index, row in input.iterrows() :
        taskRestr = []
        for hour in range(row["ReadyT"], row["Deadline"] + 1) :
            restr = "0 lt u" + str(row["User"]) + "_t" + str(row["Task"]) + "_h" + str(hour) + " lt " + str(row["MaxH"])
            taskRestr += "u" + str(row["User"]) + "_t" + str(row["Task"]) + "_h" + str(hour) + " + "
            restrictions.append(restr)

            variables.append("u" + str(row["User"]) + "_t" + str(row["Task"]) + "_h" + str(hour))
        taskRestr = "".join(taskRestr[:-2]) + "= " + str(row["EnergyD"])
        restrictions.append(taskRestr)
    
    users = []

    for i in range(1, 6) :
        rs = [s for s in restrictions if "u" + str(i) in s]
        vs = [s for s in variables if "u" + str(i) in s]
        users.append((i, rs, vs))

    resultsDf = fileToDf("TestingResults.txt")
    print(resultsDf)



