–HIGGS-REDUCED-7D.txt

```bash
spark-submit --conf spark.pyspark.python=python3 --num-executors 2 G008HW3.py /data/BDC2122/HIGGS-REDUCED-7D.txt10 150 2
spark-submit --conf spark.pyspark.python=python3 --num-executors 4 G008HW3.py /data/BDC2122/aHIGGS-REDUCED-7D.txt 10 150 4
spark-submit --conf spark.pyspark.python=python3 --num-executors 8 G008HW3.py /data/BDC2122/HIGGS-REDUCED-7D.txt 10 150 8
spark-submit --conf spark.pyspark.python=python3 --num-executors 16 G008HW3.py /data/BDC2122/HIGGS-REDUCED-7D.txt 10 150 16
```


—Artificial9000.txt

```bash
spark-submit --conf spark.pyspark.python=python3 --num-executors 2 G008HW3.py /data/BDC2122/artificial9000.txt 9 200 2
spark-submit --conf spark.pyspark.python=python3 --num-executors 4 G008HW3.py /data/BDC2122/artificial9000.txt 9 200 4
spark-submit --conf spark.pyspark.python=python3 --num-executors 8 G008HW3.py /data/BDC2122/artificial9000.txt 9 200 8
spark-submit --conf spark.pyspark.python=python3 --num-executors 16 G008HW3.py /data/BDC2122/artificial9000.txt 9 200 16
```

--run the commnad locally for the last column

```bash
python G008HW2.py 9 200
```
