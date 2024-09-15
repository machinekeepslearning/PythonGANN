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

After a set amount of time, the population will crossover their genes and create a new population
Normally, the 2 bots with the best fitness are chosen to cross over their genes/weights and reproduce, however, this destroys the genetic diversity of the population.

The weights of the neural network are considered the genes and get spliced at a random position within the weights
Genes also have a chance of mutating, turning parts of the gene into random numbers for genetic diversity

In this simulation:
2 bots are chosen in 3 different instances to create offspring
  1. first bot has the highest fitness, the second bot is a randomly selected bot (28 bots are created this way)
  2. first bot has the second highest fitness, the second bot is a randomly selected bot (42 bots are created this way)
  3. first bot has the third highest fitness, the second bot is a randomly selected bot (30 bots are created this way)

The new generation is formed by the above method + 10 bots with completely random genes/weights (Total population is 110)

Using this method of generation creation produces much greater genetic diversity, however, bots may increase their fitness much slower than the standard way due to bots with much lower fitness being allowed to pass on their genes

All of the bots spawn in a random position within a 20x20 box on the map. The position of the goal is randomized each time a generation is created to prevent similar outputs in the beginning

Goal position is randomized each time so that bots are trained to reach the goal rather than move in the same direction
