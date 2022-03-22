# Variable setup
from math import sin, cos, tan
import numpy as np
import os
import random
import matplotlib.pyplot as plt
#random.seed(891001)

#Koza Tableau
operators = ['+', '-', '*', '/', "sin", "cos"]
nums = [i/10 for i in range(1000)]
for _ in range(1000): nums.append('x')
xVals = [i/1000 for i in range(-6284, 6284)]
vals = []
mutChance = 50
crossChance = 1
popSize = 200
genNum = 100
indSize = 50
splitPercent = 1
arith = "0"

fileName = ""
# Define helper functions

# Generate random program
def randProg():
    program = []
    for _ in range(random.randrange(1, 100)):
        op = random.choice(operators)
        if op == '/': program.append((op, random.choice(nums[:1000])))
        else: program.append((op, random.choice(nums)))
    return program

def interpret(prog, x):
    total = 0
    for node in prog:
        if node[0] in ["sin", "cos"]:
            if node[0] == "sin": total = sin(total)
            elif node[0] == "cos": total = cos(total)
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
        if fileCheck: 
            buffer = abs(interpret(prog, i) - x)
            if buffer > x/10: buffer = buffer**2
            total += buffer
        else: total += (interpret(prog, x) - eval(arith)) ** 2
    total += len(prog)/5
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
                if op == '/': newProg.insert(i, (op, random.choice(nums[:1000])))
                else: newProg.insert(i, (op, random.choice(nums)))
            elif rand == 2:
                op = random.choice(operators)
                if op == '/': newProg[i] = (op, random.choice(nums[:1000]))
                else: newProg[i] = (op, random.choice(nums))
                i += 1
            elif rand == 3:
                if len(newProg) == 1: continue
                newProg.pop(i)
        else:
            i += 1
    return newProg

def xover(prog1, prog2):
    newProg = prog1.copy()
    for i in range(len(newProg)):
        if random.randint(0, 100) < crossChance:
            newProg[i] = random.choice(prog2)
    return newProg

def printProg(prog):
    output = "0"
    for node in prog:
        if node[0] in ["sin", "cos", "tan"]:
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
    if fileCheck: print("Actual fitness: ", 1/(fitness(prog, vals, fileCheck)))
    else: print("Actual fitness: ", 1/(fitness(prog, xVals, fileCheck)))
    saveCheck = input("Save best individual? (Yes/No) ").lower()
    if saveCheck in ["yes", "ye", "y"]:
        save = open("./best/" + fileName + ".csv", "w")
        for op in prog:
            save.write(op[0] + ',' + str(op[1]) + '\n')
        save.close()


    gen = [i for i in range(len(bestFits))]
    yVals = []
    zVals = []
    if fileCheck:
        for x in range(len(vals)):
            try:
                yVals.append(interpret(prog, x))
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
    plt.title("Best fitness over time")
    plt.xlabel("Generation number")
    plt.ylabel("Fitness")
    plt.legend(loc='upper center')
    plt.show()
    plt.clf()
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("Time (increment)")
    plt.ylabel("Value ($)")
    if fileCheck: 
        plt.plot(range(len(vals)), vals, color="blue", alpha=0.5, label="Dataset")
        plt.plot(range(len(vals)), yVals, color="red", alpha=0.5, label="Best program")
    else: 
        # print(zVals)
        # print(yVals)
        plt.plot(xVals, zVals, color="blue", alpha=0.5, label="Function")
        plt.plot(xVals, yVals, color="red", alpha=0.5, label="Best program")
    plt.legend(loc='upper center')
    plt.show()
    
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
    -p: Population size (Default: 100)
    -s: Percent of population chosen for Elitism (Default: 1)
    -m: Mutation chance (Default: 11)
    -c: Crossover chance (Default: 11)
    -f: csv file or function""")
args = input("Raise flags (default: -g 100 -p 100 -s 1 -m 11 -c 11 -f test.csv): ").split()

buffer = "test.csv"
fileCheck = True
equation = ""
for i in range(0, len(args), 2):
    if args[i] == "-g": genNum = int(args[i+1])
    elif args[i] == "-p": popSize = int(args[i+1])
    elif args[i] == "-s": splitPercent = float(args[i+1])
    elif args[i] == "-m": mutChance = float(args[i+1])
    elif args[i] == "-c": crossChance = float(args[i+1])
    elif args[i] == "-f": buffer = args[i+1]
    elif args[i] == "-e": equation = "true"
    
if buffer[-4:] == ".csv":
    f = open(os.path.join(os.path.dirname(__file__),buffer), "r")
    while f.readline():
        try:
            vals.append(float(f.readline().split(",")[1])) 
        except:
            break    
    fileCheck = True  
    fileName = buffer[:-4]
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
    

population = []
bestFits = []
bestProgs = []
tests = []
if len(equation) > 0: 
    for _ in range(popSize): population.append(mutate(equation))
else: 
    for _ in range(popSize): population.append(randProg())
    
for i in range(genNum):
    fitnesses = []
    if fileCheck: tests = vals[:len(vals) * 4//5]
    else: tests = random.choices(xVals, k=1000)
    print("Generation", i + 1, ":", end=" ")
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
    sumFitness = sum(fitnesses)
    bestFit = max(fitnesses)
    print("Best fitness: ", bestFit, end=", ")
    bestFits.append(bestFit)
    bestProgs.append(population[fitnesses.index(bestFit)])
    print("Best program: ", end = "")
    print("y=" + printProg(bestProgs[-1]))
    newPop = []
    split = round(popSize * (splitPercent / 100))
    for _ in range(popSize - split):
        if random.choice([True, False]):
            newProg = xover(choose(population, fitnesses, sumFitness), choose(population, fitnesses, sumFitness))
            newProg = mutate(newProg)
        else:
            newProg = mutate(choose(population, fitnesses, sumFitness))
        newPop.append(newProg)
    for _ in range(split): newPop.append(bestProgs[-1])
    population = newPop
end(bestProgs[-1])
