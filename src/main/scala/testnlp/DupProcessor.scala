package testnlp

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.math.BigDecimal.RoundingMode

object DupProcessor
{
    private val log = LoggerFactory.getLogger(getClass)

    implicit class DoubleOps(value: Double)
    {
        def round(places: Int) =
        {
            if (places < 0) throw new IllegalArgumentException
            val bd = new java.math.BigDecimal(value)
            bd.setScale(places, RoundingMode.HALF_UP).doubleValue()
        }
    }

    implicit class VectorOps(value: DenseVector)
    {
        /**
         * Second norm calculator.<br />
         * Note: This should be part of the Vector class and evaluated lazily for memoization support
         * @return
         */
        def norm =
        {
            val squaredSum = value.values.foldLeft(0.0){ case(accum, v) => accum + v * v }
            Math.sqrt(squaredSum)
        }

        def cosineDistance(other: DenseVector) =
        {
            val dotprod = value.dot(other)
            dotprod / (value.norm * other.norm)
        }
    }

    def main(args: Array[String]): Unit =
    {
        log.info("Starting DupProcessorhML application")
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("breeze").setLevel(Level.OFF)

        //val articleList = args(0)
        //val articleVectors = args(1)
        val articleVectors = "test.json"

        val sparkSession = SparkSession.builder.appName("Simple Application").master("local").getOrCreate()

        /*
        // Read the CSV file and drop the rows which can't be processed
        val articles = sparkSession.read
            .option("header", true)
            .csv(articleList)
            .filter(col("nlp") === false)
        */
        // Read the vector database
        val df = sparkSession.read.json(articleVectors)
        println(s"Read ${df.count()} rows")
        // In the database, the "d2v_vectors" column is a double array. We need to convert
        // this to a Dense vector before this can be used in the KMeans algorithm
        // Also drop the original "d2v_vectors" column
        val array2vecUDF = udf((array: Array[Double]) => { Vectors.dense(array) })
        val newdf = df
            .withColumn("vectors", array2vecUDF(col("d2v_vectors")))
            .drop("d2v_vectors")
        //println(newdf.first())
        ///*
        //val numClusters: Int = if(args.length > 2) args(3).toInt else 5000
        val numClusters = 10
        // Set up the clustering of vectors using k-means
        // Reference: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/KMeansExample.scala
        val kmeans = new KMeans().setK(numClusters).setSeed(1L).setFeaturesCol("vectors")
        val model = kmeans.fit(newdf)
        val predictions = model.transform(newdf)

        // For each cluster:
        // Compute pairwise dotproduct
        // If score > 0.95, declare duplicate, drop
        // If score > 0.80, declare same news

        // Here we need to access each element in each group. Using groupByKey
        // and mapGroups here. Other options not explored:
        // 1. groupBy with aggregator function "collect_list".
        // 2. Use Window.partitionBy.
        // Some references:
        // [1] https://stackoverflow.com/questions/49291397/spark-mapgroups-on-a-dataset
        // [2] Example of using GroupByKey in stream: https://books.japila.pl/spark-structured-streaming-internals/KeyValueGroupedDataset/
        import sparkSession.implicits._
        predictions
            .groupByKey(row => row.getInt(2))
            .mapGroups{ case(groupId, iter) =>
                // The iterator can be used only once - even if you call iter.size.
                // Let's move it to a buffer first.
                val vectorsInGroup = new ArrayBuffer[DenseVector]()
                for(a <- iter) vectorsInGroup += a.get(1).asInstanceOf[DenseVector]
                // Now compute pairwise dotproduct for the lower triangle
                val cosineDists = new ArrayBuffer[String]()
                if(vectorsInGroup.length > 2)
                {
                    for(i <- (0 until vectorsInGroup.length))
                    {
                        for(j <- 0 until i)
                        {
                            val cosineDist = vectorsInGroup(i).cosineDistance(vectorsInGroup(j))
                            cosineDists += s"($i, $j) = ${cosineDist.round(2)}"
                        }
                    }
                }
                else if(vectorsInGroup.length == 2)
                    cosineDists += s"(0, 1) = ${ vectorsInGroup(0).cosineDistance(vectorsInGroup(1)).round(2) } }"
                cosineDists
            }
            .foreach(v => println(s"${v.mkString(", ")}"))
    }
}
