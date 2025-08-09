class Solution:
    def foreignDictionary(self, words: List[str]) -> str:
        adj = {c: set() for w in words for c in w}
        
        indeg = {c: 0 for c in adj}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            min_len = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        indeg[w2[j]] += 1
                    break
        
        q = dequeue([c for c in indeg if indeg[c] == 0])
        res = []

        while q:
            c = q.popleft()
            res.append(c)
            for n in adj[c]:
                indeg[n] -= 1
                if indeg[n] == 0:
                    q.append(n)
        if len(res) != len(indeg):
            return ""
    
        return "".join(res)