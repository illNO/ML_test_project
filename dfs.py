def count_islands(matrix):
    count = 0
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                count += 1
                dfs(matrix, i, j)
    return count


def dfs(matrix, i, j):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or matrix[i][j] == 0:
        return
    matrix[i][j] = 0
    dfs(matrix, i - 1, j)
    dfs(matrix, i + 1, j)
    dfs(matrix, i, j - 1)
    dfs(matrix, i, j + 1)


# example usage
test_1 = [[0, 1, 0],
          [0, 0, 0],
          [0, 1, 1]]

test_2 = [[0, 0, 0, 1],
          [0, 0, 1, 0],
          [0, 1, 0, 0]]

test_3 = [[0, 0, 0, 1],
          [0, 0, 1, 1],
          [0, 1, 0, 1]]

print(count_islands(test_1))
print(count_islands(test_2))
print(count_islands(test_3))

