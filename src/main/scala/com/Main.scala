package com

import com.util.{ArgsParser, DataLoader, GraphUtil, Word2VecIndexGenerator}
import com.walker.DeepWalk
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main {

	def main(args: Array[String]): Unit = {

		val options = Array(
			"dataPath",
			"embeddingPath",
			"indexPath",
			"vectorSize",
			"windowSize",
			"maxIter",
			"topK",
			"numWalk",
			"walkLength",
			"masterURL"
		)

		val argParser = ArgsParser.get(args, options)

		val dataPath = argParser.getString("dataPath")
		val embeddingPath = argParser.getString("embeddingPath")
		val indexPath = argParser.getString("indexPath")
		val vectorSize = argParser.getInt("vectorSize")
		val windowSize = argParser.getInt("windowSize")
		val maxIter = argParser.getInt("maxIter")
		val numWalk = argParser.getInt("numWalk")
		val walkLength = argParser.getInt("walkLength")
		val masterURL = argParser.getString("masterURL")

		/*val dataPath = "wiki/Wiki_edgelist.txt"
		val embeddingPath = "/user/jiangpeng.jiang/data/graph_embedding/embedding_path/"
		val indexPath = "/user/jiangpeng.jiang/data/graph_embedding/index_path/"
		val vectorSize = 100
		val windowSize = 5
		val maxIter = 10
		val numWalk = 10
		val walkLength = 10
		val masterURL = "yarn-cluster"*/

		val spark = SparkSession.builder().master(masterURL).getOrCreate()
//		val spark = SparkSession.builder().getOrCreate()
//		val spark = SparkSession.builder().appName("GraphEmbedding").getOrCreate()


		val data = DataLoader.loadEdgeList(spark, dataPath)

		val processedDF = GraphUtil.preProcess(data)

		val walk = new DeepWalk()
			.setNumWalk(numWalk)
			.setWalkLength(walkLength)

		val result = walk.randomWalk(processedDF).cache()
		result.first()

		val w2v = new Word2Vec()
			.setMaxSentenceLength(100)
			.setMinCount(2)
			.setWindowSize(windowSize)
			.setVectorSize(vectorSize)
			.setMaxIter(maxIter)
			.setNumPartitions(30)
			.setInputCol("sequence")
			.fit(result)

		var wv = w2v.getVectors.toDF("word", "vector")

		wv.write.mode("overwrite").save(embeddingPath)

		wv = wv.withColumn("vector", Word2VecIndexGenerator.vecToSeq(col("vector")))

		val index = Word2VecIndexGenerator.generateIndex(20, wv, wv, "word", "vector", "word", "vector")

		index.write.mode("overwrite").save(indexPath)

	}
}