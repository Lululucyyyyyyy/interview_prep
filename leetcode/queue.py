class Queue:
    
    def __init__(self, capacity):
        # pointers for array
        self.front = 0
        self.back = 0
        self.num_elems = 0
        
        self.elems = [0]*capacity
        self.capacity = capacity
    
    def is_full(self):
        return self.num_elems == self.capacity
        
    def is_empty(self):
        return self.num_elems == 0
    
    def enqueue(self, elem):
        if self.is_full():
            new_capacity = self.capacity * 2
            new_elems = [0] * new_capacity
            if self.front == self.back == 0:
                for i in range(self.num_elems):
                    new_elems[i] = self.elems[self.front + i]
            else:
                num_front = self.capacity - self.front
                assert(self.num_elems == num_front + self.back)
                for i in range(num_front):
                    new_elems[i] = self.elems[self.front + i]
                for i in range(self.back):
                    new_elems[num_front+i] = self.elems[i]
            self.front = 0
            self.back = self.num_elems
            self.capacity = new_capacity
            self.elems = new_elems
        assert(not self.is_full())
        self.elems[self.back] = elem
        self.back += 1
        self.back %= self.capacity
        self.num_elems += 1
        
    def dequeue(self):
        assert(not self.is_empty())
        elem = self.elems[self.front]
        self.front += 1
        self.front %= self.capacity
        self.num_elems -= 1
        return elem
        
q = Queue(5)
assert(not q.is_full())
assert(q.is_empty())

q.enqueue(1)
assert(not q.is_full())
assert(not q.is_empty())

q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.enqueue(5)
assert(q.is_full())
assert(not q.is_empty())

assert(q.dequeue() == 1)
assert(q.dequeue() == 2)
assert(q.dequeue() == 3)
q.enqueue(6)
q.enqueue(7)
q.enqueue(8)
q.enqueue(9)
q.enqueue(10)
assert(q.dequeue() == 4)
assert(q.dequeue() == 5)
assert(q.dequeue() == 6)
assert(q.dequeue() == 7)
assert(q.dequeue() == 8)
assert(q.dequeue() == 9)
assert(q.dequeue() == 10)
# assert(not q.is_full())
# assert(not q.is_empty())  
     
# assert(q.dequeue() == 1)
# assert(not q.is_full())
# assert(not q.is_empty()) 
# assert(q.dequeue() == 2)
# assert(q.dequeue() == 3)
# assert(q.dequeue() == 4)
# assert(q.dequeue() == 5)
# assert(q.dequeue() == 6)
# assert(not q.is_full())
# assert(q.is_empty())      
        
        
        
        