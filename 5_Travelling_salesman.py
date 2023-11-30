import sys

def tsp_dp(graph):
    num_cities = len(graph)
    memo = {}

    # generate unique bit masks 
#for subsets of cities
    def generate_mask(subset):
        return 1 << subset

    # Recursive function to find the shortest path using DP
    def tsp_helper(current, mask):
        if mask == (1 << num_cities) - 1:  # Visited all cities
            return graph[current][0] if current != 0 else 0

        if (current, mask) in memo:
            return memo[(current, mask)]

        min_distance = sys.maxsize
        for city in range(num_cities):
            if mask & generate_mask(city) == 0:
                new_mask = mask | generate_mask(city)
                distance = graph[current][city] + tsp_helper(city, new_mask)
                min_distance = min(min_distance, distance)

        memo[(current, mask)] = min_distance
        return min_distance

    # Initialize the graph and call the recursive function
    return tsp_helper(0, 1)

graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

shortest_distance = tsp_dp(graph)
print("Shortest Distance:", shortest_distance)
