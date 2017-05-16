import numpy as np

#Sigmoid Function for non-linearity
def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradient(a):
    return (a *(1-a))

def gradientDescent(m, X, y,_y, w1, w2):
    num_iter = 50
    for i in xrange(num_iter):
        w1, w2 = backpropagation(m, X, y,_y, w1, w2)

def accuracy(_y, layer3_vector):
    arq = open('y.txt', 'w')
    arq2= open('predicted.txt', 'w')
    np.savetxt(arq,_y)
    arq.close()
    max_per_row =  np.argmax(layer3_vector, axis=1)
    max_per_row = max_per_row.reshape(5000,1)
    print(max_per_row)
    np.savetxt(arq2,max_per_row)
    arq2.close()

def main():
    #Matrix of Inputs
    X = np.genfromtxt('data_tp1.txt', delimiter=',')
    _y = np.array(X[:, 0])
    X[:, 0] = 1

    #Matrix of Real Outputs
    y = np.zeros((5000,10))
    for i in xrange(_y.size):
        y[i][int(_y[i])] = 1

    m,n = X.shape

    np.random.seed(1)

    #Matrices of Weights
    w1 = np.random.random((25,785)) # 25x785
    w2 = np.random.random((10,26)) # 10x26

    # print(_y)
    gradientDescent(m, X, y,_y, w1, w2)


def backpropagation(m, X, y, _y, w1, w2):
    #learning rate
    layer3_vector = np.zeros((5000,10))
    r = 0.5
    w1_err = np.zeros(w1.shape) 
    w2_err = np.zeros(w2.shape)
    cost = 0
    for j in xrange(m):
        #Feed Forward
        layer1 = X[j, :] ##adiciona somente uma linha
        layer2 = sigmoid(np.dot(w1, layer1)) #5000x25
        layer2 = np.append(np.ones((1,1)), layer2) #5000x26
        layer3 = sigmoid(np.dot(layer2, w2.T)) #5000x10
        layer3_vector[j, :] = layer3

        #Calculates the loss function
        cost += np.sum(-y[j, :] * np.log(layer3) - (1.0 -y[j, :]) * np.log(1.0 - layer3))

        #Calculates errors of the layers
        layer3_err = y[j, :] - layer3 # 5000x10
        w2_err += np.outer(layer3_err, layer2)

        #prints the actual error
        # print("Error: " + str(np.mean(np.abs(layer3_err))))

        layer2_err = layer3_err.dot(w2) #5000x26
        layer2_delta = layer2_err * gradient(layer2)
        w1_err += np.outer(layer2_delta[1:], layer1)

    # Takes average of error accumulated
    w1_err = w1_err/m
    w2_err = w2_err/m

    #Takes average of cost
    print(cost/m)

    w1 = w1 + w1_err*r
    w2 = w2 + w2_err*r
    
    accuracy(_y, layer3_vector)

    return w1, w2

if __name__ == '__main__':
    main()

