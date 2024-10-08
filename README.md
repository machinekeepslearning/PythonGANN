# PythonGANN
First AI project

Forgive the unreadable code. I did this when I was a lot younger and I will be creating a new version of this.

For a description of what this is:

A population of Neural Networks (which I will be referring to as "bots") is being trained to to reach a goal (green square). The closer a bot is to the goal, the better it is at the game.

This is a simulation of 110 bots using a Genetic Algorithm (simulation of natural selection).
Each bot has 5 input nodes, 4 hidden nodes, and 2 output nodes.
The input nodes consist of data relating to distance between the bot and the goal, the bot's x and y coordinates, and the goal's x and y coordinates.
The output nodes consist of actions: moving horizontally and moving vertically.
When the output node is positive, the bot moves in the positive direction (right/up) and negative direction when negative (left/down).

Fitness is measured by how close the bot is to the goal and whether or not is has passed borders of the window.
Higher fitness is indicated by a lower fitness score due to lower distance to goal being a favored trait (in retrospect this could've been done better).
If the bot passed the borders the window, then it will have 999999999 added to its fitness to deincentivize going off the map.
the fitness is proportional the distance of the bot from the goal subtracted by a goal multiplier (yeah its not really a multiplier) which is >0 when the bot is touching the goal.

After a set amount of time, the population will crossover their genes and create a new population.
Normally, the 2 bots with the best fitness are chosen to cross over their genes/weights and reproduce, however, this destroys the genetic diversity of the population.

The weights of the neural network are considered the genes and get spliced at a random position within the weights.
Genes also have a chance of mutating, turning parts of the gene into random numbers for genetic diversity

In this simulation:
2 bots are chosen in 3 different instances to create offspring
  1. first bot has the highest fitness, the second bot is a randomly selected bot (28 bots are created this way)
  2. first bot has the second highest fitness, the second bot is a randomly selected bot (42 bots are created this way)
  3. first bot has the third highest fitness, the second bot is a randomly selected bot (30 bots are created this way)

The new generation is formed by the above method + 10 bots with completely random genes/weights (Total population is 110)

Using this method of generation creation produces much greater genetic diversity, however, bots may increase their fitness much slower than the standard way due to bots with much lower fitness being allowed to pass on their genes.

All of the bots spawn in a random position within a 20x20 box on the map. The position of the goal is randomized each time a generation is created to prevent similar outputs in the beginning.

Goal position is randomized each time so that bots are trained to reach the goal rather than move in the same direction.

---------------------------------------------------------------------------------------------------------------------------

AvoidRed.py

Neural Network Architecture: 6 (input), 4 (tanh hidden), 2 (tanh output)

Inputs: whether or not each of the 4 pole rectangles are touching a red rectangle (each node has a value of 10 if touching and -10 if not touching), x and y coordinates of itself
Outputs: Left/Right, Up/Down 
  - If the first node is >0, it is moving up. down if <0
  - If the second node is >0, it is moving right. left if <0

Fitness:
Fitness is a measure of how well an entity is doing in its given environment
In this case, fitness increases when the Neural Network is alive and somewhat similar to the first project gets 99999 subtracted from its fitness when it dies to incentivize living. Fitness also increases when a pole is touching a red square to deincentivize the network from standing still and doing nothing.

Crossing Over:
Each NN has 2 sets of weights that are represented as matrices for easy feedforwarding. When crossover occurs, each matrix is flattened into a vector and 2 Neural Networks are chosen to be crossed. The first NN is randomly selected from the top 3 best performing NNs and the second NN is randomly chosen. This method encourages genetic diversity while ensuring that well performing genes are passed down. Bias vectors are crossed the same way weights are.

How the Environment Works:
There are borders at the edge of the window which will "kill" the Neural Network. There are also small red squares that travel downward that spawn at the top of the windows which will also kill the neural network upon contact with the blue body of the NN.

The Goal:
The goal of this simulation is to train Neural Networks to avoid death using sensor-like inputs which demonstrates the complexity of the behavior that AI is capable of. This was also just a cool project and incredibly fun to watch (for me at least)
