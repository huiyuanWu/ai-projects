import sys
import csv
import math
from queue import PriorityQueue
#import matplotlib.pyplot as plt

row_length = 0

#importance = [0]*row_length
all_data = []


class Node:
    def __init__(self, ty, attribute, children, classification, prob, level):
        self.name = ty  # type of node: internal or leaf
        self.attribute = attribute  # the index of the attribute, where the index is 0-41
        self.children = children
        self.classification = classification
        self.prob = prob
        self.level = level


def boolean_entropy(p, n):
    q = p / (p + n)
    #print(p, n)
    result = -1 * (q * math.log2(q) + (1-q) * math.log2(1-q))
    return result


def remainder(i, p, n, r_attributes, d_attributes):
    res = 0
    for j in range(4):
        r_attr = r_attributes[j]
        d_attr = d_attributes[j]
        pk = r_attr[i]
        nk = d_attr[i]
        res += ((pk + nk)/(p+n)) * boolean_entropy(pk, nk)
    return res


def gain(data):  # this is the importance function
    q = PriorityQueue()
    p = data[0]
    n = data[1]
    r_attributes = [data[2], data[3], data[4], data[5]]
    d_attributes = [data[6], data[7], data[8], data[9]]

    importance = [0]*row_length

    for i in range(row_length):
        importance_value = boolean_entropy(p, n) - remainder(i, p, n, r_attributes, d_attributes)
        q.put((-importance_value, i))

    for k in range(row_length):
        value = q.get()
        #print(value)
        index = value[1]

        # the most important attribute is in importance[0], say the vote to the 11th is the most important
        # then importance[0] = 11
        importance[k] = index

    return importance

#read necessary data for information gain, return them as a tuple. 
def read_data(train):
    ct_r = 0
    ct_d = 0    
    r_yea = [1]*row_length
    r_nay = [1]*row_length
    r_not_voting = [1]*row_length
    r_present = [1]*row_length
    d_yea = [1]*row_length
    d_nay = [1]*row_length
    d_not_voting = [1]*row_length
    d_present = [1]*row_length

    for row in train:
        #print(row[0])
        if row[0][row_length] == "Republican":
            ct_r += 1
            for i in range(len(row[0]) - 1):
                if row[0][i] == "Yea":
                    r_yea[i] += 1
                elif row[0][i] == "Nay":
                    r_nay[i] += 1
                elif row[0][i] == "Not Voting":
                    r_not_voting[i] += 1
                elif row[0][i] == "Present":
                    r_present[i] += 1
        elif row[0][row_length] == "Democrat":
            ct_d += 1
            for i in range(len(row[0]) - 1):
                if row[0][i] == "Yea":
                    d_yea[i] += 1
                elif row[0][i] == "Nay":
                    d_nay[i] += 1
                elif row[0][i] == "Not Voting":
                    d_not_voting[i] += 1
                elif row[0][i] == "Present":
                    d_present[i] += 1
    #print(r_present)

    return(ct_r, ct_d, r_yea, r_nay, r_not_voting, r_present, d_yea, d_nay, d_not_voting, d_present)

#read the csv to a list of list: all_data. It's global because it's widely used.
#all_data[row0, row1, ...]
def init(file_path):
    global row_length

    global all_data

    with open(file_path) as fileobj:
        read_csv = csv.reader(fileobj, delimiter=',')
        for row in read_csv:
            all_data.append(row)
            row_length = len(row)-1
            
    del all_data[0]
    #print(all_data[25][len(all_data[25])-1])

#split the complete data to 5 parts
#for example, test_1 = first part(row 1 to row 80)
#train_1 is the rest 4 parts(row 81 to row 393)
#train = (row, index of the row). Since it's going to be used as the examples of Decision_tree_learning.
def split_data():
    global all_data
    part = int(len(all_data)/5)
    test_1 = []
    train_1 = []
    test_2 = []
    train_2 = []
    test_3 = []
    train_3 = []
    test_4 = []
    train_4 = []
    test_5 = []
    train_5 = []

    for i in range(0, part, 1):
        test_1.append(all_data[i])
    for i in range(part, len(all_data), 1):
        train_1.append((all_data[i], i))
    
    for i in range(part, 2*part, 1):
        test_2.append(all_data[i])
    for i in range(0, part, 1):
        train_2.append((all_data[i], i))
    for i in range(2*part, len(all_data), 1):
        train_2.append((all_data[i], i))

    for i in range(0, 2*part, 1):
        train_3.append((all_data[i], i))
    for i in range(2*part, 3*part, 1):
        test_3.append(all_data[i])
    for i in range(3*part, len(all_data), 1):
        train_3.append((all_data[i], i))

    for i in range(0, 3*part, 1):
        train_4.append((all_data[i], i))
    for i in range(3*part, 4*part, 1):
        test_4.append(all_data[i])
    for i in range(4*part, len(all_data), 1):
        train_4.append((all_data[i], i))

    for i in range(0, 4*part, 1):
        train_5.append((all_data[i], i))
    for i in range(4*part, len(all_data), 1):
        test_5.append(all_data[i])

    return [(train_1, test_1), (train_2, test_2), (train_3, test_3), (train_4, test_4), (train_5, test_5)]
    
        


def plurality_value(egs):
    global all_data
    count_r = 0
    count_d = 0

    for i in egs:
        if all_data[i][row_length] == "Republican":
            count_r += 1
        else:
            count_d += 1

    tot = count_r + count_d
    # print("r:", count_r)
    # print("d:", count_d)
    # print("tot:", tot)
    if count_r > count_d:
        return ["Republican", count_r, count_r/tot]
    else:
        return ["Democrat", count_d, count_d/tot]


def decision_tree_learning(examples, attributes, parent_examples, level, limit):
    plurality_for_examples = None
    if len(examples) is not 0:
        plurality_for_examples = plurality_value(examples)

    values = ["Yea", "Nay", "Not Voting", "Present"]
    
    if level == limit:
        if len(examples) == 0:
            plurality_for_parent = plurality_value(parent_examples)
            return Node("leaf", None, None, plurality_for_parent[0], plurality_for_parent[2], level)

        return Node("leaf", None, None, plurality_for_examples[0], plurality_for_examples[2], level)
    
    if len(examples) == 0:  # if examples is empty then return PLURALITY-VALUE(parent examples)
        plurality_for_parent = plurality_value(parent_examples)
        return Node("leaf", None, None, plurality_for_parent[0], plurality_for_parent[2], level)
    # else if all examples have the same classification then return the classification
    elif plurality_for_examples[1] == len(examples):
        return Node("leaf", None, None, plurality_for_examples[0], plurality_for_examples[2], level)
    # else if attributes is empty then return PLURALITY-VALUE(examples)
    elif len(attributes) == 0:
        return Node("leaf", None, None, plurality_for_examples[0], plurality_for_examples[2], level)
    else:
        a = attributes.pop(0)
        #print('a', a)
        children_dict = {"Yea": None, "Nay": None, "Not Voting": None, "Present": None}
        tree = Node("internal", a, children_dict, None, None, level)
        for v in values:
            exs = []
            for i in examples:
                if all_data[i][a] == "" and v == "Nay":
                    exs.append(i)
                elif all_data[i][a] == v:
                    exs.append(i)
            subtree = decision_tree_learning(exs, attributes, examples, level+1, limit)
            tree.children[v] = subtree
        #print(tree.attribute, tree.level)
        return tree

#use the tree we trained to guess the answer. Return a list of results for every row in the test.
def guess(root, test):
    results = []
    for row in test:
        curr = root
        #print(row)
        while curr.name == "internal":
            ind = curr.attribute
            #print(ind)
            att = row[ind]
            #print(att)
            if att == "":
                att = "Nay"

            curr = curr.children[att]

        results.append(curr.classification)

    
    return results

#print the dfs tree as required format.
def print_root(root):
    print('\t'*root.level+'vote'+str(root.attribute))
    for v in root.children.values():
        if v.name == "internal":
            print_root(v)

#this function compare the guess result with the test part's classification, return the accuracy
def accuracy(result, test):
    test_class = []
    count = 0
    tot = len(result)
    for row in test:
        test_class.append(row[len(row)-1])
    #print(test_class)
    for i in range(0, tot, 1):
        if test_class[i] == result[i]:
            count += 1
    return count/tot 

#buld decision tree on train part and test it on test part, return accuracy and tree.
def train_and_test(fold, depth_limit):
    train = fold[0]
    
    test = fold[1]
    data = read_data(train)
    importance = gain(data)
    #print(importance)
    impt = []

    for y in range(row_length):
        impt.append(importance[y])

    eg = []
    for x in range(len(train)):
        eg.append(train[x][1])
        #print(train[x][1])
        
        
    result = []
    root_node = decision_tree_learning(eg, impt, None, 0, depth_limit)
    result = guess(root_node, test)
    #print(accuracy(result, test))
    print_root(root_node)
    return accuracy(result, test)

def plot_ec_2():
    pass

if __name__ == '__main__':

    train_data_path = sys.argv[1]
    depth_limit = sys.argv[2]
    depth_limit = int(depth_limit)

    accur = []
    sum = 0
    init(train_data_path)
    fold = split_data()#return the 5 folds.
    for folds in fold:
        accur.append(train_and_test(folds, depth_limit))
        print('')
    for x in accur:
        sum+=x
    accur.append(sum/5)
    strr = ""
    for i in accur:
        strr += str(i)
        strr += " "
    print(strr)

    
