package MARRIAGE_STATUS

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext

import scala.collection.mutable.ArrayBuffer

object Marriage {
  def main(args: Array[String]): Unit = {
    //--- Initialization
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext(sparkConf)
    System.setProperty("user.name", "mzsip")
    System.setProperty("HADOOP_USER_NAME", "mzsip")
    sparkConf.setAppName("YF_ALGO_MARRIAGE_MODEL") //application name
    //---
    sc.hadoopConfiguration.set("mapred.output.compress", "false")

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getRootLogger().setLevel(Level.ERROR)

    val hiveContext: HiveContext = new HiveContext(sc)
    hiveContext.setConf("mapred.output.compress", "false")
    hiveContext.setConf("hive.exec.compress.output", "false")
    hiveContext.setConf("mapreduce.output.fileoutputformat.compress", "false")
    import hiveContext.implicits._
    println("==================initSpark is OK===============")

    val today = "20170826"//调试用数据
    // val today = args(0)
    val year: Int = today.substring(0,4).trim.toInt
    val month: Int = today.substring(4,6).trim.toInt
    val day: Int = today.substring(6,8).trim.toInt
    val calendar: Calendar = Calendar.getInstance
    calendar.set(year,month-1,day)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)


    val feature_dim: Int = 30000
    val user_age_from_flyme_table_name: String = "algo.yf_user_age_collect_from_flyme"
    val user_behavoir_features_table_name: String = "algo.yf_user_behavior_features_app_install_on_30000_dims"
    val user_marriage_label_from_flyme: RDD[(String, Int)] = get_marriage_label(user_age_from_flyme_table_name, hiveContext, yestoday_Date)
    println("\n\n*********** The number of Marriage status known samples: " + user_marriage_label_from_flyme.count() + " ********\n\n")

    val data_set = get_data_set_for_build_model(user_marriage_label_from_flyme, user_behavoir_features_table_name, feature_dim, hiveContext, yestoday_Date)
    build_model(data_set, 2)

  }

  def build_model(
                   data_set: (RDD[(String, LabeledPoint)], RDD[(String, Vector)]),
                   classes_num: Int
                 ): RDD[(Double, Double)] = {
    val trainSet: RDD[(String, LabeledPoint)] = data_set._1

    val rdd_temp: Array[RDD[(String, LabeledPoint)]] = trainSet.randomSplit(Array(0.8, 0.2))
    val train_rdd: RDD[(String, LabeledPoint)] = rdd_temp(0).cache()
    val valid_rdd: RDD[(String, LabeledPoint)] = rdd_temp(1).cache()

    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(classes_num).run(train_rdd.map(_._2))
    val valid_result = valid_rdd.map(v => (model.predict(v._2.features), v._2.label))

    val multiclassMetrics = new MulticlassMetrics(valid_result)
    println("\n\n ********************** Precision = " + multiclassMetrics.precision + " ********************* \n\n")
    println("\n\n ********************** Recall = " + multiclassMetrics.recall + " ********************* \n\n")
    for (i <- 0 to classes_num-1) {
      println("\n\n ********************** Precision_" + i + " = " + multiclassMetrics.precision(i) + " ********************* \n\n")
    }

    return valid_result
  }

  def get_data_set_for_build_model(
                                    user_marriage_labeled_dataset: RDD[(String, Int)],
                                    user_behavoir_features_table_name: String,
                                    feature_dim: Int,
                                    hiveContext: HiveContext,
                                    yestoday_Date: String
                                  ): (RDD[(String, LabeledPoint)], RDD[(String, Vector)]) = {
    // get  latest date of user behavior features
    val select_latest_date_sql = "SELECT stat_date from " + user_behavoir_features_table_name + " GROUP by stat_date ORDER by stat_date DESC"
    val latest_date: String = hiveContext.sql(select_latest_date_sql).first()(0).toString
    println("\n\n ***************** The latest date of user behavior: " + latest_date.toString() + " ************* \n\n")
    // user behavior features
    val select_imei_feature_sql: String = "select * from " + user_behavoir_features_table_name + " where stat_date=" + latest_date
    val imei_feature_df: DataFrame = hiveContext.sql(select_imei_feature_sql)

    val imei_feature_rdd: RDD[(String, String)] = imei_feature_df.rdd.map(v => (v(0).toString, v(1).toString))
    println("\n\n ********************* The number of user hehavior features data: " + imei_feature_rdd.count() + " *********************** \n\n")

    // building the train and predict set
    val predict_set_rdd: RDD[(String, Vector)] = imei_feature_rdd.subtractByKey(user_marriage_labeled_dataset).map(v => {
      val imei: String = v._1
      val feature_str: String = v._2
      val features: Array[String] = feature_str.split(" ")
      val index_array: ArrayBuffer[Int] = new ArrayBuffer[Int]()
      val value_array: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      for (feature <- features) {
        val columnIndex_value: Array[String] = feature.trim.split(":")
        if (columnIndex_value.length == 2) {
          index_array += columnIndex_value(0).trim.toInt
          value_array += columnIndex_value(1).trim.toDouble
        }
      }
      (imei, index_array.toArray, value_array.toArray)
    }).filter(v => v._2.length > 0).map(v => (v._1, Vectors.sparse(feature_dim, v._2, v._3)))
    println("\n\n ********************* The size of predict set: " + predict_set_rdd.count() + " *********************** \n\n")

    val train_set_rdd: RDD[(String, LabeledPoint)] = imei_feature_rdd.join(user_marriage_labeled_dataset).map(v => {
      val imei = v._1
      val feature = v._2._1.toString
      val label = v._2._2.toString.toInt
      (imei, feature, label)
    }).map(v => {
      val imei = v._1
      val features: Array[String] = v._2.toString.split(" ")
      val label: Int = v._3
      val index_array: ArrayBuffer[Int] = new ArrayBuffer[Int]()
      val value_array: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      for (feature <- features) {
        val columnIndex_value: Array[String] = feature.trim.split(":")
        if (columnIndex_value.length == 2) {
          index_array += columnIndex_value(0).trim.toInt
          value_array += columnIndex_value(1).trim.toDouble
        }
      }
      (imei, label, index_array.toArray, value_array.toArray)
    }).filter(v => v._3.length > 0).map(v => (v._1, new LabeledPoint(v._2, Vectors.sparse(feature_dim, v._3, v._4))))
    println("\n\n ********************* The size of train set: " + train_set_rdd.count() + " *********************** \n\n")

    return (train_set_rdd, predict_set_rdd)
  }

  def get_marriage_label(
                        user_age_from_flyme_table_name: String,
                        hiveContext: HiveContext,
                        yestoday_Date: String
                        ): RDD[(String, Int)] = {
    val select_latest_date_sql = "SELECT stat_date from " + user_age_from_flyme_table_name + " GROUP by stat_date ORDER by stat_date DESC"
    val latest_date: String = hiveContext.sql(select_latest_date_sql).first()(0).toString
    println("\n\n ***************** The latest date of user_age_from_flyme: " + latest_date + " ************* \n\n")
    val select_user_age_sql: String = "select * from " + user_age_from_flyme_table_name + " where stat_date=" + latest_date
    val user_marriage_label: RDD[(String, Int)] = hiveContext.sql(select_user_age_sql).rdd.map(v => {
      val imei: String = v(0).toString
      val age: Int = v(2).toString.toInt
      var label: Int = -1
      if (age <= 35)
        label = 0
      if (age > 35)
        label = 1
      (imei, label)
    }).filter(_._2 != -1)
    return user_marriage_label
  }
}
