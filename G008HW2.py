import time
import sys
import math
from typing import List, Tuple

guesses = []


def readVectorsSeq(filename: str) -> List[Tuple[float, float]]:
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def euclidean(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res += diff*diff
    return math.sqrt(res)


def getPointCenterDistance(x: Tuple[float, float], S: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
    """
    Return center and distance.
    """
    minDist = float("inf")
    minCenter = None
    for center in S:
        currDist = euclidean(x, center)
        if currDist < minDist:
            minDist = currDist
            minCenter = center
    return minCenter, minDist


def SeqWeightedOutliers(P: List[Tuple[float, float]], W: List[float], k: int, z: int, alpha: float) -> List[Tuple[float, float]]:

    # Calculate initial guess
    r = float("inf")
    for i in range(k+z+1):
        for j in range(k+z+1):
            if i != j:
                r = min(r, euclidean(P[i], P[j]))
    r /= 2

    guesses.append(r)

    while True:
        Z = []
        for i in range(len(P)):
            Z.append({
                "coord": P[i],
                "weight": W[i]
            })
        S = []
        W_z = sum(W)

        while ((len(S) < k) and (W_z > 0)):
            maxWeight = 0
            newCenter = None
            for x in P:
                # Calculate ball weight
                ballWeight = 0
                for i in range(len(Z)):
                    if euclidean(Z[i]["coord"], x) <= (1+2*alpha)*r:
                        ballWeight += Z[i]["weight"]

                # Update best x in P
                if ballWeight > maxWeight:
                    maxWeight = ballWeight
                    newCenter = x

            # Append center
            S.append(newCenter)

            # Update Z
            Z = [point for point in Z if euclidean(
                point["coord"], newCenter) > (3+4*alpha)*r]
            W_z = sum([point["weight"] for point in Z])

        # print(f"Guess n: {len(guesses)}")
        if W_z <= z:
            return S
        else:
            r = 2*r
            guesses.append(r)


def ComputeObjective(P: List[Tuple[float, float]], S: List[Tuple[float, float]], z: int) -> float:
    assert len(P) > z, "z should be less than the amount of points of P"

    distances = []
    for x in P:
        distances.append(getPointCenterDistance(x, S)[1])

    distances = sorted(distances, reverse=True)
    return distances[z]


def main(filename: str, k: int, z: int) -> None:

    inputPoints = readVectorsSeq(filename)

    n = len(inputPoints)
    weights = [1]*n

    start_time = time.time()
    solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    end_time = time.time()

    objective = ComputeObjective(inputPoints, solution, z)

    print(f"Input size n =  {n}")
    print(f"Number of centers k =  {k}")
    print(f"Number of outliers z =  {z}")
    print(f"Initial guess =  {guesses[0]}")
    print(f"Final guess = =  {guesses[-1]}")
    print(f"Number of guesses =  {len(guesses)}")
    print(f"Objective function =  {objective}")
    print(f"Time of SeqWeightedOutliers =  {end_time - start_time}")


if __name__ == "__main__":
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python G008HW2.py <filename> <k> <z>"

    # INPUT READING

    # 1. Read the filename
    filename = sys.argv[1]

    # 2. Read number of clusters k
    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    # 2. Read number of clusters z
    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    main(filename, k, z)
