#from https://github.com/rlcode/per/blob/master/SumTree.py
#SumTree 
#a binary tree data structure where the parent's value is the
#sum of its children


import numpy

class SumTree:
    write = 0
    
    def __init__(self,capacity):
        #self.tree store the value of each node
        #only the last layer store the data
        #
        #self.data store the corresponding data
        self.capacity = capacity
        self.tree = numpy.zeros(2*capacity - 1)
        self.data = numpy.zeros(capacity,dtype=object)
        self.n_entries=0
        
    def propagate(self,idx,change):
        #update the root node
        parent = (idx - 1)//2
        self.tree[parent] += change
        
        if parent !=0:
            self.propagate(parent,change)
            
    def retrieve(self,idx,s):
        #find the sample on the leaf node
        #s is the total value
        left = 2*idx+1
        right = left+1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self.retrieve(left,s)
        else:
            return self.retrieve(right,s-self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def add(self,p,data):
        #store the priority and sample
        #only the last layer store the data
        idx = self.write + self.capacity-1
        
        self.data[self.write] = data
        self.update(idx,p)
        
        self.write+=1
        
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries+=1
     
    def update(self,idx,p):
        #update the priority
        
        change = p - self.tree[idx]
        
        self.tree[idx]=p
        self.propagate(idx,change)
        
    def get(self,s):
        #get priority and sample
        idx = self.retrieve(0,s)
        dataIndx = idx - self.capacity+1
        return (idx,self.tree[idx],self.data[dataIndx])
        
        