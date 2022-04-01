# Variable setup
from math import sin, cos, exp, log
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
#random.seed(891001)

#Koza Tableau
operators = ['+', '-', '*', '/', "sin", "cos"]
numRange = 10000
xVals = [i/1000 for i in range(-6284, 6284)]
vals = []
mutChance = 20
crossChance = 1
popSize = 200
genNum = 100
indSize = 50
splitPercent = 5
arith = "0"

fileName = ""
# Define helper functions

def randValue():
    if random.choice([True, False]): return 'x'
    else: return random.uniform(0, numRange)

# Generate random program
def randProg():
    program = []
    for _ in range(random.randrange(1, indSize)):
        op = random.choice(operators)
        if op == '/': program.append((op, randValue()))
        else: program.append((op, randValue()))
    return program

def interpret(prog, x):
    total = 0
    for node in prog:
        if node[0] not in ['+', '-', '*', '/']:
            if node[0] == "sin": total = sin(total)
            elif node[0] == "cos": total = cos(total)
            elif node[0] == "e": 
                if total < 16: total = exp(total)
                else: total = exp(16)
            elif node[0] == "log": 
                if total > 0: total = log(total)
        elif node[1] == 'x':
            if node[0] == '+': total += x
            elif node[0] == '-': total -= x
            elif node[0] == '*': total *= x
            elif node[0] == '/':
                if x != 0: total /= x
        else:
            if node[0] == '+': total += node[1]
            elif node[0] == '-': total -= node[1]
            elif node[0] == '*': total *= node[1]
            elif node[0] == '/':
                if node[1] != 0: total /= node[1]
    return total

def fitness(prog, dataset, fileCheck):
    total = 0
    for i, x in enumerate(dataset):
        if fileCheck: total += (interpret(prog, i) - x)**2
        else: total += (interpret(prog, x) - eval(arith)) ** 2
    total += len(prog)
    
    xCount = 0
    for op in prog:
        if op[1] == 'x': xCount += 1
    if xCount == 0: total += 1000000
    else: total += 1/xCount
    if total < 0: total = 1000000
    
    return total

def mutate(prog):
    newProg = prog.copy()
    i = 0
    while i < len(newProg):
        if random.randint(0, 100) < mutChance:
            rand = random.randint(1, 3)
            if rand == 1:
                if len(newProg) >= indSize: continue
                op = random.choice(operators)
                if op == '/': newProg.insert(i, (op, randValue()))
                else: newProg.insert(i, (op, randValue()))
                i += 2
            elif rand == 2:
                op = random.choice(operators)
                if op == '/': newProg[i] = (op, randValue())
                else: newProg[i] = (op, randValue())
                i += 1
            elif rand == 3:
                if len(newProg) == 1: continue
                newProg.pop(i)
        else:
            i += 1
    return newProg

def xover(prog1, prog2):
    newProg1 = prog1.copy()
    newProg2 = prog2.copy()
    if random.randint(0, 100) < crossChance:
        i = random.randint(1, min(len(prog1), len(prog2)))
        newProg1 = newProg1[:i] + prog2[i:]
        newProg2 = newProg2[:i] + prog1[i:]
    return newProg1, newProg2

def printProg(prog):
    output = "0"
    for node in prog:
        if node[0] not in ['+', '-', '*', '/']:
            output = str(node[0]) + '(' + output +')'
        else:
            output = '(' + output + str(node[0]) + str(node[1]) + ')'
    return output

def choose(population, fitnesses, sumFitness):
    chance = random.uniform(0, sumFitness)
    for i in range(len(population)):
        fit = fitnesses[i]
        if chance < fit:
            return population[i]
        else:
            chance -= fit

def end(prog):
    print("y=" + printProg(prog))
    if fileCheck: print("Actual fitness: ", (fitness(prog, vals, fileCheck)))
    else: print("Actual fitness: ", (fitness(prog, xVals, fileCheck)))

    gen = [i for i in range(len(bestFits))]
    yVals = []
    zVals = []
    if fileCheck:
        minInd, minVal = 0, 10000000000
        for x in range(len(vals)):
            try:
                value = interpret(prog, x)
                if value < minVal: 
                    minVal = value
                    minInd = x
                yVals.append(value)
            except:
                yVals.append(0)
    else:
        for x in xVals:
            try:
                yVals.append(interpret(prog, x))
            except:
                yVals.append(0)
            try:
                zVals.append(eval(arith))
            except:
                zVals.append(0)
    #for i in range(len(bestFits)): bestFits[i] = 100 / bestFits[i]
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(gen, bestFits, color="red", label="Best fitness")
    plt.plot(gen, averageFits, color="blue", label="Average fitness")
    plt.title("Best fitness over time")
    plt.xlabel("Generation number")
    plt.ylabel("Fitness")
    plt.legend(loc='best')
    plt.show()
    plt.clf()
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("Time (increment)")
    plt.ylabel("Value ($)")
    if fileCheck: 
        plt.plot(range(len(vals)), vals, color="blue", alpha=0.5, label="Dataset")
        plt.plot(range(len(vals)), yVals, color="red", alpha=0.5, label="Best program")
        
        #res = datetime.strptime(str(minInd + 83), "%j").strftime("%m-%d")
        plt.annotate(minInd, xy=(minInd, yVals[minInd]), xytext=(minInd, yVals[minInd] * 1.1), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='top')
    else: 
        # print(zVals)
        # print(yVals)
        plt.plot(xVals, zVals, color="blue", alpha=0.5, label="Function")
        plt.plot(xVals, yVals, color="red", alpha=0.5, label="Best program")
    plt.legend(loc='best')
    plt.show()
    
    saveCheck = input("Save best individual? (Yes/No): ").lower()
    if saveCheck in ["yes", "ye", "y"]:
        save = open("./best/" + fileName + ".csv", "w")
        for op in prog:
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

buffer = "test.csv"
fileCheck = True
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

if buffer[-4:] == ".csv":
    f = open(os.path.join(os.path.dirname(__file__),buffer), "r")
    while f.readline():
        try:
            time, value = f.readline().split(",")
            value = float(value)
            vals.append(value) 
        except:
            break    
    fileCheck = True  
    fileName = buffer[:-4]
    numRange = vals[0]
else: 
    arith = buffer
    fileCheck = False

if equation != "":
    load = open("./best/" + buffer, "r")
    equation = []
    for op in load:
        op = op.split(",")
        if op[1][:-1] == 'x': equation.append((op[0], op[1][:-1]))
        else: equation.append((op[0], float(op[1][:-1])))
    load.close()
    

population = []
bestFits = []
averageFits = []
bestProgs = []
tests = []
if len(equation) > 0: 
    for _ in range(popSize): population.append(mutate(equation))
else: 
    for _ in range(popSize): population.append(randProg())

prevFit = 0
for i in range(genNum):
    fitnesses = []
    if fileCheck: tests = vals[:len(vals) * 4//5]
    else: tests = random.choices(xVals, k=1000)
    print("Generation", i + 1, ":", end=" ")
    sumFitness, bestFit, bestFitIndex, minFit = 0, -1000000, 0, 1000000
    for prog in population:
        fit = fitness(prog, tests, fileCheck)
        if fit == 0:
            if fileCheck: 
                if fitness(prog, vals, fileCheck) == 0:
                    print("Perfect solution found: ", end="")
                    end(prog)
                else: fit = 1
            else:
                if fitness(prog, xVals, fileCheck) == 0:
                    print("Perfect solution found: ", end="")
                    end(prog)
                else: fit = 1
        fitnesses.append((1 / fit))
        sumFitness += fitnesses[-1]
        bestFit = max(bestFit, fitnesses[-1])
    averageFits.append(sumFitness/popSize)
    print("Best fitness:", bestFit, end=", ")
    print("Average fitness:", averageFits[-1], end=", ")
    print("Mutation chance:", str(mutChance) + "%")
    if prevFit == bestFit and mutChance < 100: mutChance += 1
    elif mutChance > minMutChance: mutChance -= 1
    prevFit = bestFit
    bestFits.append(bestFit)
    bestProgs.append(population[bestFitIndex])
    print("Best program: ", end = "")
    print("y=" + printProg(bestProgs[-1]))
    newPop = []
    split = round(popSize * (splitPercent / 100))
    newProg1, newProg2 = [], []
    count = 0
    while count < popSize - split:
        if random.choice([True, False]):
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
    for _ in range(split): newPop.append(bestProgs[-1])
    population = newPop
end(bestProgs[-1])
