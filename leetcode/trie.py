# Class Tree that support the Trie data structure and string operations
class Trie:
    def __init__(self):
        self.root = {}
        self.end_symbol = "*"

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current:
                current[char] = {}
            current = current[char]
        current[self.end_symbol] = None

    def search(self, word):
        current = self.root
        for char in word:
            if char not in current:
                return False
            current = current[char]
        return self.end_symbol in current

    def startsWith(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current:
                return False
            current = current[char]
        return True
    
    def delete(self, word):
        current = self.root
        for char in word:
            if char not in current:
                return False
            current = current[char]
        if self.end_symbol in current:
            current.pop(self.end_symbol)
            return True
        return False

    def countWordsEqualTo(self, word):
        current = self.root
        for char in word:
            if char not in current:
                return 0
            current = current[char]
        return 1 if self.end_symbol in current else 0

    def countWordsStartingWith(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current:
                return 0
            current = current[char]
        return sum(1 if self.end_symbol in child else 0 for child in current.values())

    def countWordsContainingKey(self, key):
        return self.countWordsStartingWith(key)