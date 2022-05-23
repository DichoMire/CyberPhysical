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
    #Create DataFrame from Excel file
    input = pd.read_excel("COMP3217CW2Input.xlsx")

    #Parse the input into a more convenient shape and rename columns for ease of programming
    user = input["User & Task ID"].map(lambda x: x.split("_")[0][-1])
    task = input["User & Task ID"].map(lambda x: x.split("_")[1][-1])
    task = task.replace("0", "10")

    input["User"] = user
    input["Task"] = task

    input = input.drop("User & Task ID", axis=1)
    input = input.rename(columns = {"Ready Time" : "ReadyT","Maximum scheduled energy per hour" : "MaxH", "Energy Demand" : "EnergyD"})

    #Iterate through input and collect all the variables and basic restrictions
    #A basic restriction is of the form:
    #0 lt u1_t1_h20 lt 1
    #OR
    #u1_t1_h20 + u1_t1_h21 + u1_t1_h22 + u1_t1_h23 = 1;
    variables = []
    restrictions = []

    for index, row in input.iterrows() :
        taskRestr = []
        for hour in range(row["ReadyT"], row["Deadline"] + 1) :
            userID = "u" + str(row["User"]) + "_t" + str(row["Task"]) + "_h" + str(hour)
            restr = "0 lt " + userID + " lt " + str(row["MaxH"])
            taskRestr += userID + " + "
            restrictions.append(restr)

            variables.append((row["User"], row["Task"], hour))

            #variables.append("u" + str(row["User"]) + "_t" + str(row["Task"]) + "_h" + str(hour))
        taskRestr = "".join(taskRestr[:-2]) + "= " + str(row["EnergyD"])
        restrictions.append(taskRestr)
    
    #Iterate through the restrictions and variables to create a custom data structure.
    #The data structure is a list of tuples.
    #Each tuple is of the following shape:
    #(userNumber, Restrictions, Variables)
    users = []

    for i in range(1, 6) :
        rs = [s for s in restrictions if "u" + str(i) in s]
        vs = [s for s in variables if i == int(s[0])]
        users.append((i, rs, vs))

    #Extract the TestingResults to gather the price info for the abnormal entries.
    resultsDf = fileToDf("TestingResults.txt")
    
    ex1 = resultsDf.iloc[1]

    #Since each user is separate, we do not apply game theory.
    #Therefore we simulate each user separately.
    #Iterate through each user
    for user in users :
        finRestr = "c = "
        #Iterate through each hour
        for hour in range(0, 24) :
            #Collect all variables for all the tasks for a given hour
            hours = [s for s in user[2] if hour == s[2]]
            #For each available hour that has variables
            #Append the weighted by the price variable to the final cost restriction
            for htask in hours :
                userID = "u" + str(htask[0]) + "_t" + str(htask[1]) + "_h" + str(htask[2])
                finRestr += str(ex1[hour]) + " * " + userID + " + "
        #Remove the hanging "+" sign at the end and append to list of restrictions
        finRestr = finRestr[:-3]
        user[1].append(finRestr)

    #For each user compute the program starting with the cost restriction
    for user in users :
        lpProg = "min: c;\n\n"
        lpProg += user[1][-1] + ";\n"
        #And then iterating through all basic restrictions
        for restriction in user[1][:-1] :
            lpProg += restriction.replace("lt", "<=") + ";\n"
        print(lpProg)
        
        print("\n\n\n")

    


