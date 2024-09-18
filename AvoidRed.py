import numpy as np
import pygame
import random

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

fitness_list = []
nets = []
obstacles = []
initial_left = []
initial_top = []
borders = []
velocity = 1
pop = 60
crossing = False
generation = 0
prev_gen = -1
speed = 1
timer = 20/speed

def flatten(x):
    flat = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            flat.append(x[i][j])
    return flat

def unflatten(x, a, b):
    vec = []
    mat = []
    for i in range(0, a*b, b):
        vec = []
        for j in range(b):
            vec.append(x[i + j])
        mat.append(vec)
    return mat

def drawBorders():
    for border in borders:
        pygame.draw.rect(screen, "red", border)
def spawnObstacles():
    top = 0
    for i in range(5):
        left = random.randint(0, 1280)
        initial_left.append(left)
        initial_top.append(top)
        obstacles.append(pygame.Rect(left, top, 5, 5))
def spawnNets():
    for i in range(pop):
        top = random.randint(600, 650)
        left = random.randint(50, 1250)
        fitness_list.append(0)
        nets.append(NeuralNetwork(left, top, 10, 10, i, [], []))

def globalUpdate():
    global initial_top
    global initial_left
    global nets
    for i in range(len(obstacles)):
        initial_top[i] += velocity
        pygame.draw.rect(screen, "red", obstacles[i])
        obstacles[i] = pygame.Rect(initial_left[i], initial_top[i], 5, 5)
    for i in range(pop):
        if nets[i].alive:
            nets[i].update()
    drawBorders()

def findNets(x, y):
    best_idx = 0
    next_best_idx = 0
    sorted_fitness = fitness_list.copy()
    sorted_fitness.sort(reverse=True)
    for i in range(pop):
        if nets[i].fitness == sorted_fitness[x]:
            best_idx = i
    for i in range(pop):
        if nets[i].fitness == sorted_fitness[y] and i != best_idx:
            next_best_idx = i

    return best_idx, next_best_idx

#Works as intended
def mutate(gene):
    mut_num = 4

    for i in range(len(gene)):
        if random.randint(1, 50) == mut_num:
            gene[i] = np.float64(random.uniform(-10, 10))

def cross(a, b):
    cross_num = random.randint(0, len(a)-1)
    new = a[:cross_num] + b[cross_num:]
    return new

def crossover():
    print("crossing")
    new_weight_list = []
    new_bias_list = []
    #Spawn first half of population with standard crossing


    for _ in range(pop):

        weight_mat = []
        bias_mat = []

        best_idx, next_best_idx = findNets(random.randint(0, 2), random.randint(3, pop-1))

        #best_idx, next_best_idx = findNets(0, 1)

        top_weights = nets[best_idx].flat_weights.copy()
        top_weights2 = nets[best_idx].flat_weights2.copy()
        next_weights = nets[next_best_idx].flat_weights.copy()
        next_weights2 = nets[next_best_idx].flat_weights2.copy()
        top_bias = nets[best_idx].flat_bias.copy()
        top_bias2 = nets[best_idx].flat_bias2.copy()
        next_bias = nets[next_best_idx].flat_bias.copy()
        next_bias2 = nets[next_best_idx].flat_bias2.copy()

        new_flat_weights = cross(top_weights, next_weights)
        new_flat_weights2 = cross(top_weights2, next_weights2)
        new_flat_bias = cross(top_bias, next_bias)
        new_flat_bias2 = cross(top_bias2, next_bias2)


        #Problem is above this code
        #mutate(new_flat_weights)
        #mutate(new_flat_weights2)
        #mutate(new_flat_bias)
        #mutate(new_flat_bias2)

        weight_mat.append(unflatten(new_flat_weights, 4, 6))
        weight_mat.append(unflatten(new_flat_weights2, 2, 4))
        bias_mat.append(unflatten(new_flat_bias, 4, 1))
        bias_mat.append(unflatten(new_flat_bias2, 2, 1))


        new_weight_list.append(weight_mat.copy())
        new_bias_list.append(bias_mat.copy())

        for x in new_weight_list:
            print(x)
        print("------")

    fitness_list.clear()

    for k in range(pop):
        top = random.randint(600, 650)
        left = random.randint(50, 1250)
        fitness_list.append(0)
        nets[k] = NeuralNetwork(left, top, 10, 10, k, new_weight_list[k].copy(), new_bias_list[k].copy())

#5 inputs, 4 outputs
class NeuralNetwork:
    def __init__(self, left, top, width, height, id, weights, bias):
        self.hidden = None
        self.alive = True
        self.id = id
        self.fitness = 0
        self.delta_left = 0
        self.delta_top = 0
        self.left = left
        self.top = top
        self.rect = pygame.Rect(left, top, width, height)
        self.color = "blue"
        if len(weights) == 0:
            self.weights = []
            self.weights.append(np.random.uniform(-10, 10, (4, 6)))
            self.weights.append(np.random.uniform(-10, 10, (2, 4)))
        else:
            self.weights = weights
        if len(bias) == 0:
            self.bias = []
            self.bias.append(np.random.uniform(-10, 10, (4, 1)))
            self.bias.append(np.random.uniform(-10, 10, (2, 1)))
        else:
            self.bias = bias
        self.flat_weights = flatten(self.weights[0])
        self.flat_weights2 = flatten(self.weights[1])
        self.flat_bias = flatten(self.bias[0])
        self.flat_bias2 = flatten(self.bias[1])
        self.bias = np.random.uniform(-1, 1, (2, 1))
        self.inputs = np.zeros((6, 1))
        self.outputs = []
        self.poles = []
        self.poles.append(pygame.Rect(left + width / 2, top, 10, 20))
        self.poles.append(pygame.Rect(left + width / 2, top - 10, 10, 20))
        self.poles.append(pygame.Rect(left, top + height/2, 20, 10))
        self.poles.append(pygame.Rect(left - 10, top + height/2, 20, 10))
    def update(self):
        if self.rect.collidelist(obstacles) != -1 or self.rect.collidelist(borders) != -1:
            self.fitness -= 99999
            fitness_list[self.id] = self.fitness
            self.alive = False
            return 0

        quickfit = fitness_list.copy()
        quickfit.sort(reverse=True)

        if self.fitness == quickfit[0]:
            self.color = "green"
        else:
            self.color = "blue"

        self.fitness += 0.5
        fitness_list[self.id] = self.fitness

        #check collisions
        for i in range(len(self.poles)):
            pygame.draw.rect(screen, "black", self.poles[i])
            if self.poles[i].collidelist(obstacles) != -1 or self.poles[i].collidelist(borders) != -1:
                pygame.draw.rect(screen, "red", self.poles[i])
                self.inputs[i][0] = 100
                self.fitness += 0.05
                fitness_list[self.id] = self.fitness
            else:
                self.inputs[i][0] = -100

        pygame.draw.rect(screen, self.color, self.rect)

        self.inputs[4] = self.rect.x
        self.inputs[5] = self.rect.y

        #Compute outputs and draw rects accordingly
        self.feedforward()
        self.delta_top = self.outputs[0][0]
        self.delta_left = self.outputs[1][0]
        self.top += self.delta_top
        self.left += self.delta_left
        self.rect = pygame.Rect(self.left, self.top, 10, 10)
        self.poles[0] = (pygame.Rect(self.left-5, self.top-20, 20, 20))
        self.poles[1] = (pygame.Rect(self.left-5, self.top+10, 20, 20))
        self.poles[2] = (pygame.Rect(self.left - 20, self.top-5, 20, 20))
        self.poles[3] = (pygame.Rect(self.left + 10, self.top-5, 20, 20))

    def feedforward(self):
        self.hidden = np.tanh(np.dot(self.weights[0], self.inputs) + self.bias[0])
        self.outputs = np.tanh(np.dot(self.weights[1], self.hidden) + self.bias[1])




#network = NeuralNetwork(10, 100, 10, 10, "blue")
def main():
    global running

    borders.append(pygame.Rect(0, 0, 1280, 10))
    borders.append(pygame.Rect(0, 0, 10, 720))
    borders.append(pygame.Rect(1270, 0, 10, 720))
    borders.append(pygame.Rect(0, 710, 1280, 10))

    spawnNets()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("white")

        if pygame.time.get_ticks() % 10 == 0 and not crossing:
            spawnObstacles()

        ticks = pygame.time.get_ticks()
        seconds = round(ticks/1000.0, 2)

        #if 0 <= seconds % timer <= 0.059:
        wiped = True
        for x in nets:
            if x.alive:
                wiped = False
        if wiped:
            obstacles.clear()
            initial_top.clear()
            initial_left.clear()
            crossover()
            pygame.time.delay(100)
            print(nets[0].flat_weights)
            print(nets[1].flat_weights)

        globalUpdate()

        #network.update()

        pygame.display.flip()

        clock.tick(60 * speed)



    pygame.quit()

if __name__ == "__main__":
    main()
