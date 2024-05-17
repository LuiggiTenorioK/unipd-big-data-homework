from typing import Tuple
from pyspark import SparkContext, SparkConf
import sys
import os


def product_filter(value: str, S: str = "all") -> bool:
    values = value.split(",")
    assert len(values) == 8, "Values must have 8 fields"

    quantity = int(values[3])
    country = values[7]

    return quantity > 0 and (S == "all" or S == country)


def get_product_customer_pair(value) -> Tuple[Tuple[str, int], int]:
    values = value.split(",")
    assert len(values) == 8, "Values must have 8 fields"

    product_id = values[1]
    customer_id = int(values[6])

    return ((product_id, customer_id), 1)


def gather_pairs_partitions(pairs):
    pairs_dict = {}
    for p in pairs:
        product_id, customer_id = p[0], p[1]
        if product_id not in pairs_dict.keys():
            pairs_dict[product_id] = 1
        else:
            pairs_dict[product_id] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def print_top(productPopularity, H=0):
    assert H >= 0, "H must be greather or equal than 0"

    if H == 0:
        sorted_list = sorted(productPopularity.collect(),
                             key=lambda x: x[1], reverse=True)
    else:
        sorted_rdd = productPopularity.sortBy(lambda x: -x[1])
        sorted_list = []
        count = 0
        # toLocalIterator() uses memory of the largest partition
        for pair in sorted_rdd.toLocalIterator():
            sorted_list.append(pair)
            count += 1
            if count >= H:
                break

    for pair in sorted_list:
        print("Product:", pair[0], "Popularity", pair[1], end='; ')
    print()


def run(sc: SparkContext, K: int, H: int, S: str, data_path: str) -> None:
    # Read and subdivide in K partitions
    rawData = sc.textFile(data_path, minPartitions=K).repartition(K).cache()

    # Print number of rows
    num_rows = rawData.count()
    print("Number of rows =", num_rows)

    # Transform (filter, parse and drop duplicates) the values
    productCustomer = (rawData.filter(lambda x: product_filter(x, S=S))  # Filter
                       .map(get_product_customer_pair)  # Map phase (R1)
                       # Reduce phase (R1)
                       .reduceByKey(lambda x, y: x)
                       .map(lambda x: x[0]))  # Formatting map phase

    # Print number of customers
    num_prod_cust = productCustomer.count()
    print("Product-Customer Pairs =", num_prod_cust)

    # Get pairs productCustomer with mapPartitions
    productPopularity1 = (productCustomer.mapPartitions(gather_pairs_partitions)  # Map-Reduce phase (R1)
                          .groupByKey()  # Suffle + Grouping
                          .mapValues(sum))  # Reduce phase (R2)

    # Get pairs productCustomer with map and reduceByKey
    productPopularity2 = (productCustomer.map(lambda x: (x[0], 1))  # Map phase (R1)
                          .reduceByKey(lambda x, y: x+y)  # Reduce phase (R1)
                          )

    # Print top values
    if H == 0:
        print("productPopularity1:")
        print_top(productPopularity1, H)

        print("productPopularity2:")
        print_top(productPopularity2, H)
    if H > 0:
        print("Top {} Products and their Popularities".format(H))
        print_top(productPopularity1, H)


def main(K: int, H: int, S: str, data_path: str) -> None:

    # SPARK SETUP
    conf = SparkConf().setAppName('Homework1_Group008').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    run(sc, K, H, S, data_path)


if __name__ == "__main__":
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python G008HW1.py <K> <H> <S> <file_name>"

    # INPUT READING

    # 1. Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read number of products
    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    # 3. Country parameter
    S = sys.argv[3]

    # 4. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"

    main(K, H, S, data_path)
