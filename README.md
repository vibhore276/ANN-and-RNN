# ANN-and-RNN

In the first part I implement a three layer neural network from scratch using the python's skicit-learn and numpy library.
I utilized the make_moons function of skicit-learn to create my dataset on the fly. The dataset generated has two classes, which are plotted using two colors - red and blue. The data is not linearny separable hence a general OLS wouldn't do the justice in classification. Hence, a more robust classification method - Neural Network would be required to achieve the classification goal.
I built a 3-layer neural node network with one input layer, one hidden and one output layer. For the activation function of the hidden layer I used the tanh function. Because the network needs to output the probabilities, I choose the activation function of the output layer as softmax, which is known for converting raw scores to probabilities.
The network makes prediction using the forward proporgation which is collection of matrix multiplications and the application of actiavton functions which were described above.
z1 = xW1 + b1
a1 = tanh(z1)
z2 = a1W2 + b2
a2 = Y-hat = softmax(z2)
I used the gradient descent as my loss function to measure the error in the prediction. Lastly, I analyzed and compared the loss function using different set of hidden layers in the model.
