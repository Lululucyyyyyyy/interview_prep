class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        graph = [[] for i in range(n)]

        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
        
        lows = [n]*n
        crits = []

        def dfs(x, t, p):
            if lows[x] == n:
                lows[x] = t
                for nbour in graph[x]:
                    if nbour != p:
                        ev = t + 1
                        ac = dfs(nbour, ev, x)

                        if ac >= ev:
                            crits.append((x, nbour))
                        lows[x] = min(lows[x], ac)
            return lows[x]

        dfs(n-1, 0, -1)
        return crits