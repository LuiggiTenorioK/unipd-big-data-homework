# Import Packages
from typing import List, Tuple
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math
import gc

guesses = []

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(
        lambda x: strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res += diff*diff
    return res


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res += diff*diff
    return math.sqrt(res)


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    # ------------- ROUND 1 ---------------------------
    start = time.time()
    coreset = points.mapPartitions(
        lambda iterator: extractCoreset(iterator, k+z+1))

    # END OF ROUND 1

    # ------------- ROUND 2 ---------------------------

    elems = coreset.collect()
    end = time.time()
    print("Time to compute ROUND 1: ", str((end-start)*1000), " ms")

    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # print("LEN: ",len(elems), elems)

    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    start = time.time()
    centers = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    end = time.time()
    print("Time to compute ROUND 2: ", str((end-start)*1000), " ms")

    return centers


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(
        points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[
            0]  # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point, centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point, centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P, W, k, z, alpha):
    #
    # ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    #
    # Calculate initial guess
    r = float("inf")
    for i in range(k+z+1):
        for j in range(k+z+1):
            if i != j:
                r = min(r, euclidean(P[i], P[j]))
    r /= 2

    guesses.append(r)

    gc.collect()

    while True:
        Z = []
        for i in range(len(P)):
            Z.append({
                "coord": P[i],
                "weight": W[i]
            })
        S = []
        W_z = sum(W)

        gc.collect()

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
            for i in range(len(Z)-1,-1,-1):
                point = Z[i]
                if euclidean(point["coord"], newCenter) < (3+4*alpha)*r:
                    Z.pop(i)

            # Z = [point for point in Z if euclidean(
            #     point["coord"], newCenter) > (3+4*alpha)*r]
            W_z = sum([point["weight"] for point in Z])

            gc.collect()

        # print(f"Guess n: {len(guesses)}")
        if W_z <= z:
            return S
        else:
            r = 2*r
            guesses.append(r)

        gc.collect()


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
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


def ComputeDistances(P: List[Tuple[float, float]], S: List[Tuple[float, float]], z: int) -> float:
    # Return z+1 farthest distances
    distances = []
    for x in P:
        distances.append(getPointCenterDistance(x, S)[1])

    distances = sorted(distances, reverse=True)
    return distances[0:z+1]

def computeObjective(points, centers, z):
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#
    distances = points.mapPartitions(
        lambda iterator: ComputeDistances(iterator, centers, z))

    elems = distances.collect() # len = (z+1)*L
    elems = sorted(elems, reverse=True)
    # print("ELEMS: ", elems)
    return elems[z] # Discard z distances, return z+1 th elem

# Just start the main program
if __name__ == "__main__":
    main()
