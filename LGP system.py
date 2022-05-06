# Library setup
from math import sin, cos, exp, log, sqrt, tan
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
from heapq import heapify, heappop, heappush

# Seeding for testing purposes
# random.seed(891001)

# Koza Tableau
operators = ['+', '-', '*', '/', "sin", "cos", "tan", "e", "log"]
numRange = 10000
xVals = [i/1000 for i in range(-6284, 6284)]
vals, vals2 = [], []
mutChance = 20
crossChance = 80
popSize = 200
genNum = 100
indSize = 100
splitPercent = 5
arith = "0"
tourSize = 2

# Miscellaneous variables
fileName = ""

# Define helper functions

# Choose a random value or 'x'
def randValue():
    if random.choice([True, False]): return 'x'
    else: return random.uniform(0, numRange)

# Generate random program
def randProg():
    program = []
    for _ in range(random.randrange(1, indSize)): # Guarantee at least one instruction per individual
        op = random.choice(operators)
        program.append([op, randValue()])
    program = [random.uniform(0, numRange), program]
    return program

# Evaluate x against an individual
def interpret(prog, x, setNum):
    total = prog[0]
    for node in prog[1]:
        if node[0] not in ['+', '-', '*', '/']: # Handle operators that ignore operands
            if node[0] == "sin": total = sin(total)
            elif node[0] == "cos": total = cos(total)
            elif node[0] == "tan": total = tan(total)
            elif node[0] == "e": # Protection against int overflow
                if total < 16: total = exp(total)
                else: total = exp(16)
            elif node[0] == "log": # Protection against -INF
                if total > 0: total = log(total)
        elif node[1] == 'x':
            if node[0] == '+': total += x
            elif node[0] == '-': total -= x
            elif node[0] == '*': total *= x
            elif node[0] == '/': # Protection against INF
                if x != 0: total /= x
        else:
            if node[0] == '+': total += node[1]
            elif node[0] == '-': total -= node[1]
            elif node[0] == '*': total *= node[1]
            elif node[0] == '/': # Protection against INF
                if node[1] != 0: total /= node[1]
    if setNum == 0: total += norm1
    else: total += norm2
    return total

# Fitness function
def fitness(prog, dataset, fileCheck, dataset2 = []):
    total = 0
    for i, x in enumerate(dataset):
        # Current fitness function is based on Root Mean Squared Error (RMSE)
        if fileCheck: 
            buffer = interpret(prog, i, 0)
            if buffer < 0: total += 1000000 
            else: total += (buffer - x)**2
        else: total += (interpret(prog, x, 0) - eval(arith)) ** 2
    total += len(prog)
    
    xCount = 0
    for op in prog[1]:
        if op[0] in ['+', '-', '*', '/'] and op[1] == 'x': xCount += 1
    if xCount == 0: total += 1000000
    else: total += 1/xCount # Minor bias towards functions that utilize more X's
    total = round(sqrt(total/indSize), 6)
    
    if multi:
        total2 = 0
        for i, x in enumerate(dataset2):
            buffer = interpret(prog, i, 1)
            if buffer < 0: total2 += 1000000 
            else: total2 += (buffer - x)**2
        total2 += len(prog)
        
        xCount = 0
        for op in prog[1]:
            if op[0] in ['+', '-', '*', '/'] and op[1] == 'x': xCount += 1
        if xCount == 0: total2 += 1000000
        else: total2 += 1/xCount # Minor bias towards functions that utilize more X's
        total2 = round(sqrt(total2/indSize), 6)
        
        total = min(total, total2)
    
    return total

# Mutate individual
def mutate(prog):
    newProg = prog[1].copy()
    i = 0
    if random.randint(0, 100) < mutChance: prog[0] = random.uniform(0, numRange)
    while i < len(newProg):
        if random.randint(0, 100) < mutChance: # Uniform mutation dependent on dynamic mutation chance
            rand = random.randint(1, 3) # Equal opportunity for addition, deletion, or mutation
            if rand == 1:
                if len(newProg) >= indSize: continue
                op = random.choice(operators)
                if op == '/': newProg.insert(i, [op, randValue()])
                else: newProg.insert(i, [op, randValue()])
                i += 2
            elif rand == 2:
                op = random.choice(operators)
                if op == '/': newProg[i] = [op, randValue()]
                else: newProg[i] = [op, randValue()]
                i += 1
            elif rand == 3:
                if len(newProg) == 1: continue
                newProg.pop(i)
        else:
            i += 1
    return [prog[0], newProg]

# One-point crossover operator that returns two children
def xover(prog1, prog2):
    newProg1 = prog1[1].copy()
    newProg2 = prog2[1].copy()
    if random.randint(0, 100) < crossChance:
        i = random.randint(1, min(len(prog1), len(prog2)))
        newProg1 = newProg1[:i] + prog2[1][i:]
        newProg2 = newProg2[:i] + prog1[1][i:]
    return [prog1[0], newProg1], [prog2[0], newProg2]

# Interpret individuals into human readable formulas
def printProg(prog):
    output = str(prog[0])
    for node in prog[1]:
        if node[0] not in ['+', '-', '*', '/']:
            output = str(node[0]) + '(' + output +')'
        else:
            output = '(' + output + str(node[0]) + str(node[1]) + ')'
    return output

# Choose individual from population with chance proportional to fitness value
def choose(population, fitnesses, sumFitness):
    tournament = random.sample([i for i in range(len(population))], tourSize)
    contPop = []
    contFit = []
    for contest in tournament:
        contPop.append(population[contest])
        contFit.append(fitnesses[contest])
    return random.choices(contPop, weights=reversed(contFit), k = 1)[0]

# Eliminate adjacent operations to improve runtime
def checkAdjacent(prog):
    index = 0
    while index < len(prog[1]) - 1:
        if prog[1][index][0] == "+" and prog[1][index + 1][0] == "+" or prog[1][index][0] == "-" and prog[1][index + 1][0] == "-": 
            if prog[1][index][1] != "x" and prog[1][index + 1][1] != "x":
                prog[1][index][1] += prog[1][index + 1][1]
                prog[1].pop(index)
            else: index += 1
        elif prog[1][index][0] == "+" and prog[1][index + 1][0] == "-" or prog[1][index][0] == "-" and prog[1][index + 1][0] == "+":
            if prog[1][index][1] != "x" and prog[1][index + 1][1] != "x":
                prog[1][index][1] -= prog[1][index + 1][1]
                prog[1].pop(index)
            else: index += 1
        elif prog[1][index][0] == "*" and prog[1][index + 1][0] == "*" or prog[1][index][0] == "/" and prog[1][index + 1][0] == "/":
            if prog[1][index][1] != "x" and prog[1][index + 1][1] != "x":
                prog[1][index][1] *= prog[1][index + 1][1]
                prog[1].pop(index)
            else: index += 1
        elif prog[1][index][0] == "*" and prog[1][index + 1][0] == "/" or prog[1][index][0] == "/" and prog[1][index + 1][0] == "*":
            if prog[1][index][1] != "x" and prog[1][index + 1][1] != "x":
                prog[1][index][1] /= prog[1][index + 1][1]
                prog[1].pop(index)
            else: index += 1
        else: index += 1
    return prog

# Terminate program
def end(prog):
    print("y=" + printProg(prog))
    accuracy = 0
    if fileCheck: 
        if multi: accuracy = (fitness(prog, vals, fileCheck, vals2))/goatFit
        else: accuracy = (fitness(prog, vals, fileCheck))/goatFit
    else: accuracy = (fitness(prog, xVals, fileCheck))/goatFit

    if accuracy != 1: accuracy = abs(goatFit - accuracy)/goatFit
    accuracy = str(accuracy * 100) + "%"

    print("Accuracy: ", accuracy)

    gen = [i for i in range(len(bestFits))]
    yVals, yVals2 = [], []
    zVals = []
    if fileCheck:
        minInd, minVal = 0, 10000000000
        for x in range(len(vals) + 180):
            try:
                value = interpret(prog, x, 0)
                if value < minVal: 
                    minVal = value
                    minInd = x
                yVals.append(value)
            except:
                yVals.append(0)
            if multi:
                try:
                    value = interpret(prog, x, 1)
                    yVals2.append(value)
                except:
                    yVals2.append(0)
    else:
        for x in xVals:
            try:
                yVals.append(interpret(prog, x, 0))
            except:
                yVals.append(0)
            try:
                zVals.append(eval(arith))
            except:
                zVals.append(0)
    #for i in range(len(bestFits)): bestFits[i] = 100 / bestFits[i]
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(gen, bestFits, color="red", label="RMSE")
    #plt.plot(gen, averageFits, color="blue", label="Average fitness")
    plt.title("Least RMSE over time")
    plt.xlabel("Generation number")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    plt.show()
    plt.clf()
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("Time (increment)")
    plt.ylabel("Value ($)")
    if fileCheck: 
        plt.plot(range(len(vals)), vals, color="blue", alpha=0.5, label="Dataset")
        if multi: plt.plot(range(len(vals)), vals2, color="green", alpha=0.5, label="Dataset 2")
        plt.plot(range(len(vals) + 180), yVals, color="red", alpha=0.5, label="Best program")
        if multi: plt.plot(range(len(vals) + 180), yVals2, color="red", alpha=0.5, label="Best program norm adjusted")
        
        #res = datetime.strptime(str(minInd + 83), "%j").strftime("%m-%d")
        plt.annotate(minInd, xy=(minInd, yVals[minInd]), xytext=(minInd, yVals[minInd] * 1.1), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='top')
    else: 
        # print(zVals)
        # print(yVals)
        plt.plot(xVals, zVals, color="blue", alpha=0.5, label="Function")
        plt.plot(xVals, yVals, color="red", alpha=0.5, label="Best program")
    plt.legend(loc='best')
    plt.show()
    
    if fileCheck and multi:
        plt.plot(range(len(vals)), vals2, color="blue", alpha=0.5, label="Dataset")
        plt.plot(range(len(vals) + 180), yVals2, color="red", alpha=0.5, label="Best program")
        plt.annotate(minInd, xy=(minInd, yVals[minInd]), xytext=(minInd, yVals[minInd] * 1.1), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='top')
        plt.legend(loc='best')
        plt.show()
    
    # Save individual as csv file to be loaded in the future
    saveCheck = input("Save best individual? (Yes/No): ").lower()
    if saveCheck in ["yes", "ye", "y"]:
        save = open("./best/" + fileName + ".csv", "w")
        for op in prog[1]:
            save.write(op[0] + ',' + str(op[1]) + '\n')
        save.close()
    
    exit()

def compare(eq):
    yVals = []
    for x in range(len(vals)):
        try:
            eq = printProg(prog)
            yVals.append(eval(eq))
        except:
            yVals.append(0)
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(range(len(vals)), vals, color="blue", alpha=0.5, label="Dataset")
    plt.plot(range(len(vals)), yVals, color="red", alpha=0.5, label="Best program")
    plt.legend(loc='upper center')
    plt.show()
    

# Main function
print("""
    Flags available:
    -g: Number of generations (Default: 100)
    -p: Population size (Default: 200)
    -s: Percent of population chosen for Elitism (Default: 5)
    -m: Minimum mutation chance (Default: 20)
    -c: Crossover chance (Default: 100)
    -f: csv file or function
    -e: Load existing model""")
args = input("Raise flags (default: -g 100 -p 200 -s 5 -m 20 -c 50 -f test.csv): ").split()

# Check for raised flags
buffer = "gold.csv"
fileCheck = True
multi = False
norm1, norm2 = 0, 0
equation = ""
for i in range(0, len(args)):
    if args[i] == "-g": genNum = int(args[i+1])
    elif args[i] == "-p": popSize = int(args[i+1])
    elif args[i] == "-s": splitPercent = float(args[i+1])
    elif args[i] == "-m": mutChance = float(args[i+1])
    elif args[i] == "-c": crossChance = float(args[i+1])
    elif args[i] == "-f": buffer = args[i+1]
    elif args[i] == "-e": equation = "true"
minMutChance = mutChance
split = round(popSize * (splitPercent / 100))
# Read data
if buffer[-4:] == ".csv":
    f = open(os.path.join(os.path.dirname(__file__),buffer), "r")
    check = f.readline().split(",")
    if len(check) == 3: multi = True
    while f:
        data = f.readline().split(",")
        if len(data) < 2: break
        value = float(data[1])
        vals.append(value)
        if multi: 
            value = float(data[2])
            vals2.append(value)
    fileCheck = True  
    fileName = buffer[:-4]
    numRange = vals[0] # Scale available values to the initial value of dataset
    norm1 = numRange
    if multi: norm2 = vals2[0]
else: 
    arith = buffer
    fileCheck = False

# Load existing individual
if equation != "":
    load = open("./best/" + buffer, "r")
    equation = []
    for op in load:
        op = op.split(",")
        if op[1][:-1] == 'x': equation.append([op[0], op[1][:-1]])
        else: equation.append([op[0], float(op[1][:-1])])
    load.close()
    equation = [norm1, equation]
    
population = []
bestFits = []
tests, tests2 = [], []
goats = []
goat = []
goatFit = 10000000

# Populate initial generation
if len(equation) > 0: 
    # Delta of loaded individual
    for _ in range(popSize): population.append(checkAdjacent(mutate(equation)))
else: 
    # Random individuals
    for _ in range(popSize): population.append(checkAdjacent(randProg()))

prevFit = 0
if fileCheck: 
    tests = vals[:(len(vals) * 4)//5] # Reserve 90% of dataset as training data\
    if multi:
        tests2 = vals2[:(len(vals) * 4)//5]
else: tests = random.choices(xVals, k=1000)

for i in range(genNum):
    fitnesses = []
    bestProgs = []
    heapify(bestProgs)
    print("Generation", i + 1, ":", end=" ")
    sumFitness, bestFit = 0, 1000000
    for index, prog in enumerate(population):
        fit = fitness(prog, tests, fileCheck, tests2)
        if fit == 0 or mutChance == 100:
            if fileCheck: 
                if fitness(prog, vals, fileCheck) == 0: 
                    print("Perfect solution found: ", end="")
                    end(prog)
                else: 
                    print("Prolonged stagnation detected")
                    end(goat)
            else:
                if fitness(prog, xVals, fileCheck) == 0: 
                    print("Perfect solution found: ", end="")
                    end(prog)
                else: 
                    print("Prolonged stagnation detected")
                    end(goat)
        
        # Reserving best programs for elitism
        if split > 0:
            if len(bestProgs) >= split:
                temp = bestProgs[0]
                if -temp[0] > fit: 
                    heappush(bestProgs, (-fit, index))
                    heappop(bestProgs)
            else: heappush(bestProgs, (-fit, index))
        
        fitnesses.append(fit)
        sumFitness += fit
        bestFit = min(bestFit, fitnesses[-1])
        if bestFit < goatFit:
            goatFit = bestFit
            goat = prog
    averageFit = (sumFitness/popSize)
    print("Best fitness:", bestFit, end=", ")
    print("Average fitness:", averageFit, end=", ")
    print("Mutation chance:", str(mutChance) + "%")
    if prevFit == bestFit and mutChance < 100: mutChance += 1
    else: mutChance = minMutChance
    prevFit = bestFit
    bestFits.append(bestFit)
    goats.append(goat)
    print("Best program: ", end = "")
    print("y=" + printProg(goat))
    newPop = []
    newProg1, newProg2 = [], []
    count = 0
    while count < popSize - split:
        if random.choice([True, False]) and count != popSize - split - 1: # Equal chance between reproduction or delta
            newProg1, newProg2 = xover(choose(population, fitnesses, sumFitness), choose(population, fitnesses, sumFitness))
            newProg1 = mutate(newProg1)
            newProg2 = mutate(newProg2)
            newPop.append(newProg1)
            newPop.append(newProg2)
            count += 2
        else:
            newProg1 = mutate(choose(population, fitnesses, sumFitness))
            newPop.append(newProg1)
            count += 1
    while len(bestProgs) > 0: 
        try:
            newPop.append(population[heappop(bestProgs)[1]]) # Elitism
        except:
            newPop.append(newPop[-1])
        count += 1
    population = newPop
end(goat)
