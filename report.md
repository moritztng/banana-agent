# Training Details
## Neural Network Architecture

![Network](https://i.ytimg.com/vi/MItCZ6GK2JM/maxresdefault.jpg)

I used a Dueling Neural Network for this MDP. Unlike the image, I did not use convolutions, since we are not operating on raw pixel data. 
The input layer has 37 dimensions, since we have a state length of 37. After that two hidden layers with 64 dimensions each are following. 
I used ReLU as the activation function for all hiden layers. Finally we have a hidden layer with 32 dimensions for the action values and another one with 32 dimensions for the state value. 
Because we have a action size of 4, the output layer of the action values has 4 dimensions and the output layer for the state values has 1 dimension. 

Since we are making use of Deep-Q-Learning we have two instances of these networks. 

## Loss Function 
As the objective for learning the action-value-function i.e. the Q-Network we use the Mean Squared Error Loss. 
That makes sense, because we use non-linear regression to approximate the action-value-function. 

## Optimizer 
As the optimizer we use the Adam optimizer, because it is convenient, common and state of the art. 

## Experience Replay 
As expected in Deep-Q-Learning we use experience replay to break the correlations between the `(S,A,R,S)` tuples.
I had not to limit the replay buffer by a fixed buffer size, because I solved the MDP in 500 Episodes, which led to total buffer size of only 90K. 
I implemented Prioritized Experience Replay, but could not benefit from it. Further it resulted in a significant increase in computing cost, because we had to make two forward passes for each experience tuple. 
Hence, I disabled it. 

## Double-Q and fixed weights
To break the correlation between target and prediction in the loss, I made use of two Networks. 
One for learning the action-value-function and one with fixed weights for the policy. 
As it is common in Double-Q-Learning the target is calculated using both networks.

## Policy 
We use a epsilon-greedy-policy, which is based on the action-values of the network with the fixed weights. 
We start with `epsilon=1`, so the actions are equally distributed. With each episode epsilon is multiplied with a fixed decay rate until it reaches a fixed minimum value.

## Future Work 
I am not hopeless, that it is possible to make the agent benefit from prioritized experience replay. So I would definetely continue finetuning the hyperparameters to make it work. 
In general I am convinced, that it is possible to solve the MDP in about 300 episodes by further finetuning the hyperparameters. 
Eventuelly I could automate the finetuning process by using some meta-learning algorithms like population-based learning. 
