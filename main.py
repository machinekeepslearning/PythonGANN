import math

import numpy
import numpy as np
import random, pygame, sys

speed_modifier = 1
size = w, h = 1000, 800
yellow = 120, 120, 0
black = 0, 0, 0
red = 255, 0, 0
blue = 0, 0, 255
gray = 128, 128, 128
green = 0, 255, 0
orange = 255, 165, 0
nets = []
n_nets = 100
n_rand_nets = 10
globalfitness = []
highestfitness = 0
clock = pygame.time.Clock()
timer = 10
generation = 0
crossing = False # If the timer isnt working try using this


#-----------------------------------------------------------------------------------------------------------------------

'''
Inputs for the neural net:
- distance from the sprite to the goal
- sprite x position
- sprite y position
- goal x position
- goal y position

input matrix shape = 5 rows 1 column
first weight matrix shape = 4 rows 5 columns
hidden matrix shape = 4 rows 1 column
second weight matrix shape = 2 rows 4 columns
output matrix shape = 2 rows 1 column

np.dot(weight matrix, other matrix)


1 hidden layers: 
4 neurons

2 outputs:
x movement
y movement
'''


# ----------------------------------------------------------------------------------------------------------------------

class Block(pygame.sprite.Sprite):

    def __init__(self, color, width, height):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        self.rect = self.image.get_rect()


class Goal(Block):
    def __init__(self, color, width, height):
        super(Goal, self).__init__(color, width, height)
        self.rect.x = random.randint(850, 900)
        self.rect.y = random.randint(650, 700)
        self.x = self.rect.x
        self.y = self.rect.y


class Player(Block):

    def __init__(self, color, width, height):
        super(Player, self).__init__(color, width, height)
        self.speed = 1
        self.x = self.rect.x
        self.y = self.rect.y

    def update(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[pygame.K_RIGHT]:
            self.x += self.speed
            self.rect.topleft = round(self.x, 2), round(self.y, 2)
        if key_pressed[pygame.K_LEFT]:
            self.x -= self.speed
            self.rect.topleft = round(self.x, 2), round(self.y, 2)
        if key_pressed[pygame.K_UP]:
            self.y -= self.speed
            self.rect.topleft = round(self.x, 2), round(self.y, 2)
        if key_pressed[pygame.K_DOWN]:
            self.y += self.speed
            self.rect.topleft = round(self.x, 2), round(self.y, 2)


# -----------------------------------------------------------------------------------------------------------------------


def activation_h(x):
    return max(0.1 * x, x)


Lrelu_vec = np.vectorize(activation_h)


class NeuralNetwork(Block):
    def __init__(self, weight_i, weight_h, idx, goal_x, goal_y, color):
        super().__init__(color, 10, 10)
        self.color = color
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.touched_goal = False
        self.outside_map = False
        self.goal_multiplier = 0
        self.w_i = weight_i
        self.w_h = weight_h
        self.idx: int = int(idx)
        self.rect.topleft = [random.randint(200, 220), random.randint(200, 220)]
        #self.rect.topleft = [200, 200]
        self.x = self.rect.x
        self.y = self.rect.y
        self.hidden = None
        self.output = None
        self.sums_i = None
        self.sums_h = None
        self.bias = None
        self.fitness = 0
        self.dist = math.sqrt((goal_y - self.y) ** 2 + (goal_x - self.x) ** 2)
        self.inputs = np.array([[self.x],
                                [self.y],
                                [self.dist],
                                [goal_x],
                                [goal_y]])

        if self.w_i is None:
            self.init_weights()
            #print(self.w_i)
            #print(self.w_h)
        if self.bias is None:
            self.bias = np.random.randint(-10, 10, (4, 1))

    def init_weights(self):
        self.w_i = np.random.uniform(-1, 1, (4, 5))
        self.w_h = np.random.uniform(-1, 1, (2, 4))

    def updateInputs(self):
        self.dist = math.sqrt((self.goal_y - self.y) ** 2 + (self.goal_x - self.x) ** 2)
        self.inputs = np.array([[self.x],
                                [self.y],
                                [self.dist],
                                [self.goal_x],
                                [self.goal_y]])
    def feedforward(self):
        self.sums_i = np.dot(self.w_i, self.inputs)
        self.hidden = np.tanh(self.sums_i + self.bias)
        self.sums_h = np.dot(self.w_h, self.hidden)
        self.output = np.tanh(self.sums_h)

    def update(self):
        #if self.idx == 0:
        #    print(self.inputs)
        self.outside_map = self.rect.x > 1000 or self.rect.x < 0 or self.rect.y > 800 or self.rect.x < 0
        self.updateInputs()
        self.feedforward()
        self.x = round(self.x + self.output[0][0], 5)
        self.y = round(self.y + self.output[1][0], 5)
        self.rect.topleft = self.x, self.y

        if self.outside_map:
            self.fitness = 999999999 + self.dist*.1 - self.goal_multiplier
        elif self.rect.colliderect(plr.rect):
            self.goal_multiplier = 10
            self.fitness = self.dist * .1 - 100 - self.goal_multiplier
        elif not self.outside_map and not self.rect.colliderect(plr.rect):
            self.fitness = self.dist*.1 - self.goal_multiplier
        globalfitness[self.idx] = self.fitness
        testfit = globalfitness.copy()
        testfit.sort()
        if self.fitness == testfit[0] or self.fitness == testfit[1]:
            self.image.fill(blue)
        else:
            self.image.fill(self.color)

def updateGame():
    screen.fill(yellow)
    plr.update()
    if not crossing:
        for n in nets:
            screen.blit(n.image, n.rect)
            n.update()

    screen.blit(goal.image, goal.rect)
    screen.blit(plr.image, plr.rect)
    pygame.display.flip()


plr = Player(black, 10, 10)
goal = Goal(green, 10, 10)

for i in range(n_nets + n_rand_nets):
    globalfitness.append(0.0)
    nets.append(NeuralNetwork(None, None, idx=i, goal_x=goal.rect.x, goal_y=goal.rect.y, color=red))

pygame.init()

screen = pygame.display.set_mode(size)

def mutation(matrix, m_type):
    if m_type == "i":
        row = random.randint(0, 3)
        randindex = random.randint(0, 4)
        matrix[row][randindex] = random.uniform(-1, 1)
    else:
        row = random.randint(0, 1)
        randindex = random.randint(0, 3)
        matrix[row][randindex] = random.uniform(-1, 1)
def crossover(bestNet, secondBestNet):
    global crossing

    crossing = True
    #pygame.time.wait(1000) #Use if nets are none
    '''bestNet = None
    secondBestNet = None'''
    f_newW_i = None
    f_newW_h = None
    s_newW_i = None
    s_newW_h = None

    minCross = 1
    maxCross = 4

    #Find the best nets
    '''sortedFit = globalfitness.copy()
    sortedFit.sort()
    FirstFit = sortedFit[0]
    SecondFit = sortedFit[1]

    for net in nets:
        if net.fitness == FirstFit:
            print("Best Net Found!")
            bestNet = net
        if net.fitness == SecondFit:
            print("Second Best Net Found!")
            secondBestNet = net
    if bestNet is None:
        print('Cant find best net')
    if SecondFit is None:
        print("cant Find second best net")'''

    #print(sortedFit)
    #print(bestNet.idx)
    #print(secondBestNet.idx)

    # Actual crossover for first weight set
    for q in range(4):
        selectIndex = random.randint(minCross, maxCross)
        firstWeights = bestNet.w_i[q][:selectIndex]
        secondWeights = secondBestNet.w_i[q][selectIndex:]
        newRow = [np.append(firstWeights, secondWeights)]

        AfirstWeights = bestNet.w_i[q][selectIndex:]
        AsecondWeights = secondBestNet.w_i[q][:selectIndex]
        AnewRow = [np.append(AsecondWeights, AfirstWeights)]


        if f_newW_i is None:
            f_newW_i = newRow.copy()
        else:
            f_newW_i = numpy.append(f_newW_i, newRow, 0)

        if s_newW_i is None:
            s_newW_i = AnewRow.copy()
        else:
            s_newW_i = numpy.append(s_newW_i, AnewRow, 0)
        #print(selectIndex)
        #print(newW_i)

    # Actual crossover for second weight set
    for b in range(2):
        selectIndex = random.randint(minCross, maxCross)
        firstWeights = bestNet.w_h[b][:selectIndex]
        secondWeights = secondBestNet.w_h[b][selectIndex:]
        newRow = [np.append(firstWeights, secondWeights)]

        AfirstWeights = bestNet.w_h[b][selectIndex:]
        AsecondWeights = secondBestNet.w_h[b][:selectIndex]
        AnewRow = [np.append(AsecondWeights, AfirstWeights)]

        if f_newW_h is None:
            f_newW_h = newRow.copy()
        else:
            f_newW_h = numpy.append(f_newW_h, newRow, 0)

        if s_newW_h is None:
            s_newW_h = AnewRow.copy()
        else:
            s_newW_h = numpy.append(s_newW_h, AnewRow, 0)

    #print(bestNet.w_i)
    #print(secondBestNet.w_i)
    #print(newW_i)
    #print(newW_h)

    #mutation

    mutationRate = random.randint(1, 10)

    if mutationRate == 1 or mutationRate == 2:
        mutation(f_newW_i, "i")
        mutation(f_newW_h, "h")

    #pygame.time.wait(100)

    return f_newW_i, f_newW_h, s_newW_i, s_newW_h

def findNets(fits, firstFit, secondFit):
    global bestNet, secondNet
    for net in nets:
        if net.fitness == fits[firstFit]:
            # print("Best Net Found!")
            bestNet = net
        if net.fitness == fits[secondFit]:
            # print("Second Best Net Found!")
            secondNet = net
    return bestNet, secondNet

while True:
    clock.tick(120 * speed_modifier)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    updateGame()
    ticks = pygame.time.get_ticks()
    seconds = round(ticks/1000, 2)

    if timer * 0.988 <= seconds % timer <= timer * 1.012 and seconds != 0 and crossing == False:
        goal.rect.x = random.randint(100, 900)
        goal.rect.y = random.randint(100, 700)
        #pygame.time.wait(100)
        generation += 1
        print(f"Generation: {generation}")
        crossing = True
        sortedFit = globalfitness.copy()
        sortedFit.sort()

        arr_of_newW_i = []
        arr_of_newW_h = []
        arr_of_s_newW_i = []
        arr_of_s_newW_h = []

        '''
        FirstFit = sortedFit[0]
        SecondFit = sortedFit[1]

        #Find The best 2 nets
        for net in nets:
            if net.fitness == FirstFit:
                #print("Best Net Found!")
                bestNet = net
            if net.fitness == SecondFit:
                #print("Second Best Net Found!")
                secondBestNet = net
        
        if bestNet is None:
            print('Cant find best net')
            print(globalfitness)
            for net in nets:
                print(net.fitness == FirstFit)
        if SecondFit is None:
            print("cant Find second best net")
            print(globalfitness)
            for net in nets:
                print(net.fitness == FirstFit)
        '''

        #pygame.time.wait(1000)
        print("Crossover will now happen!")
        testfit = globalfitness.copy()
        testfit.sort()
        print("Best Fitness of last gen: " + str(testfit[0]))
        for i in range(int(n_nets/2)):
            if i < 14:
                bestNet, secondNet = findNets(testfit, 0, random.randint(0, n_nets - 1))
            if i < 21:
                bestNet, secondNet = findNets(testfit, 1, random.randint(0, n_nets - 1))
            else:
                bestNet, secondNet = findNets(testfit, 2, random.randint(0, n_nets - 1))
            f_new_i, f_new_h, s_new_i, s_new_h  = crossover(bestNet, secondNet)

            arr_of_newW_i.append(f_new_i)
            arr_of_newW_h.append(f_new_h)
            arr_of_s_newW_i.append(s_new_i)
            arr_of_s_newW_h.append(s_new_h)
        print(len(arr_of_newW_i))
        for _ in range(n_nets + n_rand_nets):
            del nets[0]
        #print(nets)
        for i in range(int(n_nets/2)):
            nets.append(NeuralNetwork(arr_of_newW_i[i], arr_of_newW_h[i], i, goal.rect.x, goal.rect.y, gray))
        for i in range(int(n_nets/2)):
            nets.append(NeuralNetwork(arr_of_s_newW_i[i], arr_of_s_newW_h[i], i+n_nets/2, goal.rect.x, goal.rect.y, orange))
        for i in range(n_rand_nets):
            nets.append(NeuralNetwork(None, None, n_nets + i, goal.rect.x, goal.rect.y, red))
        print(len(nets))
        pygame.time.wait(100)
        #print("before purge: " + str(nets))
        #print("after purge: " + str(nets))
    crossing = False

# crossover set up:

'''
When generation dies, sort the fitness table and collect the best candidates.
Then collect the weights of each candidate and initialize random integers to define sections of the weights to crossover

How the crossover will work 
    -Each Row must crossover
    -Choose random index for each row
    -Crossover portions of row using the index
'''
