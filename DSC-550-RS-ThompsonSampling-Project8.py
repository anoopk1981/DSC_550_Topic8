# Thompson Sampling for Slot Machines

# Importing the libraries


#The library is extremely useful when working with lists and multi-dimensional arrays.
import numpy as np
import pandas as pd
# Setting conversion rates and the number of samples
# Conversion rates are the winning chances.
# Slot machine 1 have a 15% chance of win.
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]

#N is the number of samples in the predefined data set.
#d is the number of slot machines.
N = 10000
d = len(conversionRates)

# Creating the dataset
#The dataset have the predefined set of wins and losses for every slot machine for every sample.
# The below code wll give a set that will tell us if at somestep i we have won or not by playing a certain slot machine.
# A 2D array of full of zeros, of size N * d is created.
# First for loop iterates through the rows and second for loop iterates through the columns
# We check each slot machine (each column) to see if a random float number from range (0,1) is smaller than the slot machine's conversion rate.
# As an example, Since there is an equal chance of getting any float number from the range (0,1), the chances of getting a number smaller than x (where x is also in range (0,1)) are equal to x.
# Slot machine 1 has a 15% chance of returning a high reward for d = 0.15, which means that 15 times in 100 it will get a smaller float number than 0.15.
# In other words, the smaller the random float number, the better your chances of winning.
# Therefore, if one of the N samples from your dataset X is as follows: [0, 1, 0, 0, 1], you would win at that time by playing slot machine number 2 or number 5.

X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

#Write dataset to a csv so as to use for visualization
df = pd.DataFrame(X)
df.to_csv('dataset.csv', index = False)

# It is necessary to create two arrays that will count how many times you have lost and won on each slot machine
# Making arrays to count our losses and wins
nPosReward = np.zeros(d) #Store number of wins
nNegReward = np.zeros(d) # Store number of losses

# Start a for loop that iterates through every sample in our dataset and chooses the best slot machine.
# Initially, only create two variables, one called selected, which will tell you which slot machine was chosen, and maxRandom, which you will use to get the highest Beta distribution guess across all slot machines

# Now we take random guesses from our Beta distribution and find the highest value across all your slot machines.
# We can use np.random.beta(a,b) function in numpy to do this.
# Taking our best slot machine through beta distibution and updating its losses and wins
# We create a for loop to iterate through every slot machine and find the best one.
# For each slot machine of index j (remember that we are still in the bigger for loop with index i), we take a random draw, called randomBeta, from our Beta distribution, and check if it is greater than maxRandom.
#
# If it is, then we reassign maxRandom to be equal to randomBeta, and set selected to be equal to the index of this new highest-guess slot machine j.
# It is also worth mentioning what the a and b arguments of the Beta function are in this case; they're the number of wins and losses we've had on the specific slot machine.
# Bbigger the first argument, the better, and the higher our random guess will be; the bigger the second argument, the worse, and the lower our random guess will be.
#Once we have selected the best slot machine, we update the nPosReward and nNegReward variables.

for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# Showing which slot machine is considered the best
# We add together nPosReward and nNegReward variables to display how many times each slot machine was chosen by the algorithm.
nSelected = nPosReward + nNegReward 
for i in range(d):
    print('Machine number ' + str(i + 1) + ' was selected ' + str(nSelected[i]) + ' times')
print('Conclusion: Best machine is machine number ' + str(np.argmax(nSelected) + 1))
