import numpy as np

y = open("y.txt")
predicted = open("predicted.txt")
eOutput = np.loadtxt(y)
rOutput = np.loadtxt(predicted) 

#Compares both matrices
accuracy = 0
for i in xrange(5000):
    if (eOutput[i] == rOutput[i]):
        accuracy+=1
print(str((accuracy/float(5000))*100) + " '%' de acerto") 
y.close()
predicted.close()

