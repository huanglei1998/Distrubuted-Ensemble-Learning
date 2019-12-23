//package mllib.DecisionTree

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.DecisionTree
import org.joda.time.format._
import org.joda.time._
import org.joda.time.Duration



object BinaryDecesionTree {
  def setLogger ={
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress","false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }

  def PrepareData(spark:SparkSession):(RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint],Map[String,Int])={

    val rawDataWithHeader = spark.sparkContext.textFile("/dsjxtjc/2019211331/classify/train.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{(idx,iter)=>if(idx==0) iter.drop(1) else iter}
    val lines = rawData.map(_.split("\t"))
    println("共计："+lines.count()+"条")

    val categoriesMap = lines.map(fields=>fields(3)).distinct.collect.zipWithIndex.toMap
    val labeledPointRDD = lines.map { fields =>
        val trFields = fields.map(_.replaceAll("\"",""))
        val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
        val index = categoriesMap(fields(3))
        categoryFeaturesArray(index) = 1
        val numericalFeatures = trFields.slice(4,fields.size-1).map(d=>if(d=="?") 0.0 else d.toDouble)
        val label = trFields(fields.size-1).toInt
        LabeledPoint(label,Vectors.dense(categoryFeaturesArray++numericalFeatures))
    }

    val Array(trainData,validationData,testData) = labeledPointRDD.randomSplit(Array(0.8,0.1,0.1))
    return (trainData,validationData,testData,categoriesMap)
  }

  def trainModel(trainData:RDD[LabeledPoint],impurity:String,maxDepth:Int,maxBins:Int):(DecisionTreeModel,Double)={
    val startTime = new DateTime()
    val model = DecisionTree.trainClassifier(trainData,2,Map[Int,Int](),impurity,maxDepth,maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime,endTime)
    (model,duration.getMillis())
  }

  def evaluateModel(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):Double ={
    val scoreAndLabel = validationData.map{data =>
      val score = model.predict(data.features)
      (score,data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabel)
    val AUC = Metrics.areaUnderROC()
    AUC
  }

  def trainEvaluate(trainData:RDD[LabeledPoint],validationData:RDD[LabeledPoint]):DecisionTreeModel= {
    println("训练开始>>>>")
    val (model,time) = trainModel(trainData,"entropy",10,10)
    println("训练完成，所需时间:"+time+"毫秒")
    val AUC = evaluateModel(model,validationData)
    println("评估结果AUC="+AUC)
    return model

  }

  def PredictData(spark: SparkSession, model: DecisionTreeModel, categoriesMap: Map[String, Int])={
    val rawDataWithHeader = spark.sparkContext.textFile("/dsjxtjc/2019211331/classify/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{(idx,iter)=>if(idx==0) iter.drop(1) else iter}
    val lines = rawData.map(_.split("\t"))
    println("共计:"+lines.count.toString()+"条")
    val dataRDD = lines.take(20).map{fields =>
      val trFields = fields.map(_.replaceAll("\"",""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val index = categoriesMap(fields(3))
      categoryFeaturesArray(index) = 1
      val numericalFeatures = trFields.slice(4,fields.size).map(d => if(d=="?") 0.0 else d.toDouble)
      val label = 0
      val url = trFields(0)
      val Features = Vectors.dense(categoryFeaturesArray++numericalFeatures)
      val predict = model.predict(Features).toInt
      var predictDesc = {predict match{
        case 0=>"暂时性网页(ephemeral)";
        case 1=>"长青网页(evergreen)"
      }}
      println("网址:"+url+"==>"+predictDesc)
    }
  }
  def main() = {
    setLogger
    val spark = SparkSession.builder().appName("BinaryDecisionTree").getOrCreate()
    println("RunDecisionTreeBinary")
    println("===============数据准备阶段===============")
    val (trainData,validationData,testData,categoriesMap) = PrepareData(spark)
    trainData.persist();validationData.persist();testData.persist();
    println("===============训练评估阶段===============")
    val model = trainEvaluate(trainData,validationData)
    println("===============测试阶段===============")
    val auc = evaluateModel(model,testData)
    println("===============预测数据===============")
    PredictData(spark,model,categoriesMap)

  }
}

BinaryDecesionTree.main()
