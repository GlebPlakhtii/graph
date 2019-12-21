import numpy as np


class Vertex:
    def __init__(self, vertex, number=None, selected=False, prev=None):
        self.vertex = vertex
        self.selected = selected
        self.number = number
        self.prev = prev


class Edge:
    def __init__(self, v1, v2, weight=float('inf')):
        self.edge = [v1, v2]
        self.weight = weight


class Graph:
    def __init__(self, vertex_numb, vertexes, file=None):
        self.vertex_numb = vertex_numb
        self.vertexes = vertexes
        self.edges = []
        self.matrix = []
        self.edges_numb = 0
        if file:
            self.__read_adj_matrix(file)
            self.__get_edges()

    def __read_adj_matrix(self, file):
        with open(file, 'r') as f:
            self.matrix = f.read().split()
            for j in range(self.vertex_numb ** 2):
                if self.matrix[j] == 'i':
                    self.matrix[j] = float('inf')
                else:
                    self.matrix[j] = float(self.matrix[j])
            self.matrix = np.array(self.matrix, float).reshape((self.vertex_numb, self.vertex_numb))

    def show_matrix(self):
        print(self.matrix)

    def __get_edges(self):
        for i in range(self.vertex_numb):
            for j in range(self.vertex_numb):
                if self.matrix[i][j] < float('inf'):
                    self.edges.append(Edge(self.vertexes[i], self.vertexes[j], self.matrix[i][j]))

                    self.edges_numb += 1

    def __get_weight(self, v1, v2):
        for edge in self.edges:
            if edge.edge[0] == v1 and edge.edge[1] == v2:
                return edge.weight
        return float('inf')

    def show_edges(self):
        for edge in self.edges:
            print('(' + edge.edge[0].vertex + ' , ' + edge.edge[1].vertex + ')' + ' : ' + str(edge.weight))

    def get_adj_list(self, ver):
        adj_list = []
        for i in range(self.vertex_numb):
            if self.matrix[ver.number][i] < float('inf'):
                adj_list.append(self.vertexes[i])
        return adj_list

    def bfs(self, f_ver):
        for ver in self.vertexes:
            ver.selected = False
            ver.prev = None
        way = []
        f_ver.selected = True
        raw_ver = [f_ver]
        while raw_ver:
            ver = raw_ver[0]
            way.append(ver)

            del raw_ver[0]
            for v in self.get_adj_list(ver):
                if not v.selected:
                    v.prev = ver
                    v.selected = True
                    raw_ver.append(v)
        return way

    def dfs(self, f_ver):
        for ver in self.vertexes:
            ver.selected = False
            ver.prev = None
        way = []
        f_ver.selected = True

        raw_ver = [f_ver]
        while raw_ver:
            ver = raw_ver[-1]
            way.append(ver)
            del raw_ver[-1]
            for v in self.get_adj_list(ver):
                if not v.selected:
                    v.prev = ver
                    v.selected = True
                    raw_ver.append(v)
        return way

    def show_way(self, way):
        way = list(map(lambda a: a.vertex, way))
        route = ''
        for i in range(len(way)):
            route += way[i]
            if i < len(way) - 1:
                route += ' -> '
        print(route)

    def warshal(self):
        matrixes = []
        for i in range(self.vertex_numb):
            matrixes.append(np.zeros((self.vertex_numb, self.vertex_numb), int))
        matrixes[0] = self.matrix
        for k in range(1, self.vertex_numb):
            for i in range(self.vertex_numb):
                for j in range(self.vertex_numb):
                    if matrixes[k - 1][i][j] or (matrixes[k - 1][i][k] and matrixes[k - 1][k][j]) < float('inf'):
                        matrixes[k][i][j] = 1

        for i in range(self.vertex_numb):
            matrixes[-1][i][i] = 1
        return matrixes[-1]

    def floyd_warshal(self):
        mtrx = self.matrix

        for k in range(1, self.vertex_numb):
            for i in range(self.vertex_numb):
                for j in range(self.vertex_numb):
                    mtrx[i][j] = min(mtrx[i][j], mtrx[i][k], mtrx[k][j])
        return mtrx

    def dijkstra(self, v0):
        for ver in self.vertexes:
            ver.prev = None

        row_vertexes = [v0]
        min_way = {}

        for v in self.vertexes:
            min_way[v] = float('inf')

        min_way[v0] = 0
        current = v0

        while set(row_vertexes) != set(self.vertexes):
            for u in self.get_adj_list(current):
                if u not in row_vertexes and min_way[u] >= min_way[current] + self.__get_weight(current, u):
                    min_way[u] = min_way[current] + self.__get_weight(current, u)
                    u.prev = current
            argmin = float('inf')
            for ver in self.vertexes:
                if ver not in row_vertexes and min_way[ver] < argmin:
                    argmin = min_way[ver]
                    current = ver

            row_vertexes.append(current)

        return min_way

    def prim(self):
        spanning_tree = Graph(self.vertex_numb, self.vertexes)
        raw_vertexes = self.vertexes.copy()

        proc_vertexes = [self.vertexes[0]]
        raw_vertexes.remove(self.vertexes[0])
        min_edge = None
        while raw_vertexes:
            min_weight = float('inf')
            for raw_ver in raw_vertexes:
                for proc_ver in proc_vertexes:
                    if self.__get_weight(proc_ver, raw_ver) <= min_weight:
                        min_weight = self.__get_weight(proc_ver, raw_ver)
                        min_edge = Edge(proc_ver, raw_ver, min_weight)

            if min_weight <= float('inf'):
                spanning_tree.edges.append(min_edge)
                raw_vertexes.remove(min_edge.edge[1])
                proc_vertexes.append(min_edge.edge[1])

            else:
                print('incoherent graph')
                return 5

        return spanning_tree

    
grp = Graph(8, [Vertex('a', 0), Vertex('b', 1), Vertex('c', 2), Vertex('d', 3), Vertex('e', 4), Vertex('f', 5),
                Vertex('g', 6), Vertex('h', 7)], 'input.txt')

grp.show_matrix()

grp.show_edges()

grp.show_way(grp.bfs(grp.vertexes[0]))
dij = grp.dijkstra(grp.vertexes[0])
for ver in grp.vertexes:
    print(grp.vertexes[0].vertex + ',' + ver.vertex + ' : ' + str(dij[ver]))

print('\n\n\n\n')
tree = grp.prim()
tree.show_edges()
