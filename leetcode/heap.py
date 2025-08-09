class Heap:
    def __init__(self):
        self.heap = []

    def parent(self, x):
        return (x - 1) // 2

    def left_child(self, x):
        return x * 2 + 1

    def right_child(self, x):
        return x * 2 + 2

    def compare(self, x, y):
        return self.heap[x] < self.heap[y]

    def heapify_up(self, x):
        if x == 0:
            return
        p = self.parent(x)
        if self.compare(x, p):
            self.heap[x], self.heap[p] = self.heap[p], self.heap[x]
            self.heapify_up(p)

    def heapify_down(self, x):
        l = self.left_child(x)
        r = self.right_child(x)
        s = x
        if l < len(self.heap) and self.compare(l, s):
            s = l
        if r < len(self.heap) and self.compare(r, s):
            s = r
        if s != x:
            self.heap[x], self.heap[s] = self.heap[s], self.heap[x]
            self.heapify_down(s)
        
    def insert(self, x):
        self.heap.append(x)
        self.heapify_up(len(self.heap) - 1)
    
    def pop(self):
        if self.heap:
            ret = self.heap[0]
            self.heap[0] = self.heap[-1]
            self.heap.pop()
            self.heapify_down(0)
            return ret