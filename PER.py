import random
import numpy as np

'''
Priority Experience with unsorted sum tree backend

Methods:
1) sample: Get the batch given a batch size 'n'. Returns tuple (indx, data of size n)

2) add: adds error and associated sample to memory

3) update: update the sampled memory given errors at the indices
'''


class PriorityMemory:

    eps = 0.01                      #small constant to ensure non zero priority
    alpha = 0.6                     # Controls difference between high/low error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, error, sample):
        priority = self.get_priority(error)
        self.tree.add(priority, sample)

    def sample(self, n):
        batch = []
        indxs = []
        segment = self.tree.total() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i+1)

            s = random.uniform(a, b)
            (indx, priority, data) = self.tree.get(s)
            batch.append((indx, data))

        return batch

    def update(self, indx, error):
        priority = sel.get_priority(error)
        self.tree.update(indx, priority)

class SumTree:

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def propagate(self, indx, delta):
        parent = (indx - 1) // 2

        self.tree[parent] += delta

        if parent != 0:                     #Keep propagating up the tree the delta
            self.propagate(parent, delta)

    def retrieve(self, indx, s):
        left = 2*indx + 1
        right = left + 1

        if left >= len(self.tree):
            return indx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        indx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(indx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, indx, priority):
        delta = priority - self.tree[indx]

        self.tree[indx] = priority
        self.propagate(indx, delta)

    def get(self, s):
        indx = self.retrieve(0, s)
        dataIndx = indx - self.capacity + 1

        return (indx, self.tree[indx], self.data[dataIndx])
