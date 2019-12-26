import sys, math
from queue import Queue, PriorityQueue
import heapq
import numpy as np

searched = set()
frontier_sum = 0
expanded_sum = 0
path_cost = 0

def parse(path):
    temp = []
    temp_list = []
    with open(path) as fp:
        for line in fp:
            line = line.strip('\n').split(" ")
            temp.append(line)
    return temp

def matrix_to_list(matrix):
    temp_list = []
    for line in matrix:
        for x in line:
            temp_list.append(x)
    return temp_list

def print_sol(node, root):
    trace = []
    cur = node
    global path_cost

    while cur.parent != None:
        trace.append(cur.matrix)
        cur = cur.parent

    trace.append(root.matrix)
    pos = []
    trace = trace[::-1]
    
    for matrix in trace:
        tup = findBlank(matrix)
        pos.append(tup)

    for i in range(len(pos)-1):
        last_x = pos[i][0]
        last_y = pos[i][1]
        new_x = pos[i+1][0]
        new_y = pos[i+1][1]
        #print(trace[i][new_x][new_y])
        path_cost += int(trace[i][new_x][new_y])

    str = ""
    for matrix in trace:
        str1 = ""
        for line in matrix:
            s = ""
            for i in line:
                s += i
                s += " "
            str += s
            str += "\n"
        str1 += "\n"
        str += str1
    str = str[:-1]
    print(str)
           

#return the number of inversion find in matrix. 
def check_inversion(array):
    count = 0
    for i in range(0, 8):
        for j in range(i+1, 9):
            if array[i] != '.' and array[j] != '.':
                if int(array[i]) > int(array[j]):
                    count = count + 1
    return count

#check the number of inversions in the matrix, if the number of inversion is even, it is solvable.
def check_solvable(array):
    return check_inversion(array) % 2 == 0

def is_solved(matrix):
    array = matrix_to_list(matrix)
    temp = ['.', '1', '2', '3', '4', '5', '6', '7', '8']
    return array == temp
    

def copy_matrix(matrix):
    new_matrix = [[0 for x in range(3)] for y in range(3)]
    
    for i in range(0, 3):
        for j in range(0, 3):
            new_matrix[i][j] = matrix[i][j]
    
    return new_matrix


def findBlank(mat):
    for i in range(0, 3):
        for j in range(0, 3):
            if mat[i][j] == '.':
                return [i, j]

def get_path_cost(node, root):
    trace = []
    cur = node
    cost = 0

    while cur.parent != None:
        trace.append(cur.matrix)
        cur = cur.parent

    trace.append(root.matrix)
    pos = []
    trace = trace[::-1]
    
    for matrix in trace:
        tup = findBlank(matrix)
        pos.append(tup)

    for i in range(len(pos)-1):
        last_x = pos[i][0]
        last_y = pos[i][1]
        new_x = pos[i+1][0]
        new_y = pos[i+1][1]
        #print(trace[i][new_x][new_y])
        cost += int(trace[i][new_x][new_y])
    
    return cost

def manhattan_distance(matrix):
    h = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if matrix[i][j] != '.':
                num = int(matrix[i][j])
                correctX = math.floor(num / 3)
                correctY = num % 3
                h += abs(correctX - i) + abs(correctY - j)
    return h

class Node:
    def __init__(self, matrix, parent, cost):
        self.matrix = matrix
        self.parent = parent
        self.cost = cost
    def __lt__(self, other):
        return self.cost < other.cost
    


frontier = []
def greedy(matrix):
    initial_cost = manhattan_distance(matrix)
    root = Node(matrix, None, initial_cost)
    global frontier_sum
    global expanded_sum
    heapq.heappush(frontier, root)
    frontier_sum += 1

    while len(frontier) != 0:
        node = heapq.heappop(frontier)
        expanded_sum += 1
        if is_solved(node.matrix):
            #print(node.matrix)
            print_sol(node, root)
            return

        searched.add(' '.join(matrix_to_list(node.matrix)))

        next_states = getNexts(node)

        for x in next_states:
            #print(x.matrix)
            heapq.heappush(frontier, x)
            #print(frontier_sum)
            frontier_sum += 1

def getNexts(matNode):
    mat = matNode.matrix
    arrO = matrix_to_list(mat)
    invArrO = check_inversion(arrO)
    tup = findBlank(mat)
    x = tup[0]
    y = tup[1]
    nextArr = []
    if canMoveL(x, y, mat):
        matL = moveL(x, y, mat)
        arrL = matrix_to_list(matL)
        invArrL = check_inversion(arrL)
        if not invArrL > invArrO:
            cost = manhattan_distance(matL)
            nodeL = Node(matL, matNode, cost)
            nextArr.append(nodeL)
    if canMoveR(x, y, mat):
        matR = moveR(x, y, mat)
        arrR = matrix_to_list(matR)
        invArrR = check_inversion(arrR)
        if not invArrR > invArrO:
            cost = manhattan_distance(matR)
            nodeR = Node(matR, matNode, cost)
            nextArr.append(nodeR)
    if canMoveU(x, y, mat):
        matU = moveU(x, y, mat)
        arrU = matrix_to_list(matU)
        invArrU = check_inversion(arrU)
        if not invArrU > invArrO:
            cost = manhattan_distance(matU)
            nodeU = Node(matU, matNode, cost)
            nextArr.append(nodeU)
    if canMoveD(x, y, mat):
        matD = moveD(x, y, mat)
        arrD = matrix_to_list(matD)
        invArrD = check_inversion(arrD)
        if not invArrD > invArrO:
            cost = manhattan_distance(matD)
            nodeD = Node(matD, matNode, cost)
            nextArr.append(nodeD)
    return nextArr

def canMoveL(x, y, mat):
    if y != 0:
        temp = moveL(x, y, mat)
        temp_str = ' '.join(matrix_to_list(temp))
        if temp_str in searched:
            return False
        else:
            return True
    else:
        return False

def moveL(x, y, mat):
    toReturn = copy_matrix(mat)
    temp = toReturn[x][y-1]
    toReturn[x][y-1] = "."
    toReturn[x][y] = temp
    return toReturn

def canMoveR(x, y, mat):
    if y != 2:
        temp = moveR(x, y, mat)
        temp_str = ' '.join(matrix_to_list(temp))
        if temp_str in searched:
            return False
        else:
            return True
    else:
        return False

def moveR(x, y, mat):
    toReturn = copy_matrix(mat)
    temp = toReturn[x][y+1]
    toReturn[x][y+1] = "."
    toReturn[x][y] = temp
    return toReturn

def canMoveU(x, y, mat):
    if x != 0:
        temp = moveU(x, y, mat)
        temp_str = ' '.join(matrix_to_list(temp))
        if temp_str in searched:
            return False
        else:
            return True
    else:
        return False

def moveU(x, y, mat):
    toReturn = copy_matrix(mat)
    temp = toReturn[x-1][y]
    toReturn[x-1][y] = "."
    toReturn[x][y] = temp
    return toReturn

def canMoveD(x, y, mat):
    if x != 2:
        temp = moveD(x, y, mat)
        temp_str = ' '.join(matrix_to_list(temp))
        if temp_str in searched:
            return False
        else:
            return True
    else:
        return False

def moveD(x, y, mat):
    toReturn = copy_matrix(mat)
    temp = toReturn[x+1][y]
    toReturn[x+1][y] = "."
    toReturn[x][y] = temp
    return toReturn

        
if __name__ == '__main__':
    path = sys.argv[1]
    search = sys.argv[2]
    matrix = parse(path)
    array = matrix_to_list(matrix)
    
