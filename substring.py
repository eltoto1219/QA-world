import numpy as np

def substring(x, y):
    m = len(x)
    n = len(y)
    x = [c for c in x]
    y = [c for c in y]
    c = np.zeros((m+1, n+1))
    seq = ""
    result = 0
    for i in range(m+1):
        old = len(seq)
        for j in range(n+1):
            if i == 0 or j == 0 :
                c[i][j] = 0
            elif x[i-1] == y[j-1]:
                c[i][j] = c[i-1][j-1]+1
                if len(seq) == old and c[i][j] > result:
                    seq += y[j-1]
                    #row col for storing max val
                    row = i
                    col = j
                result = max(result, c[i][j])
            else:
                c[i][j] = 0
    assert len(seq) == result
    return c, seq, result, (row, col)

#official method to retrieve the longest common substring
# if true, then no common substring exists
def getLCS(X, c, length, row, col):
    X = [x for x in X]
    length = int(length)
    if length == 0:
        #print("No Common Substring")
        return
    result = ""
    while c[row][col] != 0:
        length -= 1
        result += X[row -1]
        row -= 1
        col -= 1
    #print("last longest substring: (official)", result[::-1])
    return result[::-1]

#for printing results
def printSStr(c, seq, ans):
    c = c[1:, 1:]
    print("c matrix:")
    for s in c:
        print("".join([str(l)[0]+"," for l in s][:-1]))
    print("the first longest seg (un-official): {}".format(seq))
    print("length : {}".format(ans))

if __name__ == "__main__":
    x = "HBCDGH"
    y = "BAEDFH"
    c, seq, length, (row, col) = substring(x, y)
    printSStr(c, seq, length)
    getLCS(x, c, length, row, col)
