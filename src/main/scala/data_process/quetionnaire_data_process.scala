package data_process

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext

import scala.collection.mutable.ArrayBuffer

object quetionnaire_data_process {
  def main(args: Array[String]): Unit = {
    //--- Initialization
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext(sparkConf)
    System.setProperty("user.name", "mzsip")
    System.setProperty("HADOOP_USER_NAME", "mzsip")
    sparkConf.setAppName("YF_ALGO_QUESTIONNAIRE_DATA_PROCESS") //application name
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

    val today = "20171225"
    //调试用数据
    //val today = args(0)
    val year: Int = today.substring(0, 4).trim.toInt
    val month: Int = today.substring(4, 6).trim.toInt
    val day: Int = today.substring(6, 8).trim.toInt
    val calendar: Calendar = Calendar.getInstance
    calendar.set(year, month - 1, day)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)
    println("\n\n******************* yestoday_Date: " + yestoday_Date + " ********************\n\n")
    get_questionnaire_data_new(hiveContext, yestoday_Date)
  }

  def get_questionnaire_data_new(hiveContext: HiveContext, yestoday_Date: String) = {
    val questionnaire_select_sql: String = "SELECT flymeid, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12 from ext_metis.dwm_log_questionnaire"
    val questionnaire_data = hiveContext.sql(questionnaire_select_sql)
    println("\n\n******************* questionnaire_data: " + questionnaire_data.count() + " ******************")

    val select_imei_uid_sql: String = "SELECT imei,uid from  user_profile.edl_device_uid_mz_rel where stat_date=" + yestoday_Date
    val imei_uid = hiveContext.sql(select_imei_uid_sql)
    val questionnaire_refined_df = questionnaire_data.join(imei_uid, questionnaire_data("flymeid") === imei_uid("uid"))
    println("******************* questionnaire_refined_df(key from flymeid to imei): " + questionnaire_refined_df.count() + " ****************\n\n")

    // write DB
    questionnaire_refined_df.registerTempTable("temp")
    val questionnaire_table_name: String = "algo.yf_questionnaire_data_new"
    val creat_sql: String = "create table if not exists " + questionnaire_table_name + " (flymeid string, Q1 string, Q2 string, Q3 string, Q4 string, Q5 string, Q6 string, Q7 string, Q8 string, Q9 string, Q10 string, Q11 string, Q12 string, imei string, uid string) stored as textfile"
    val insert_sql: String = "insert overwrite table " + questionnaire_table_name + " select * from temp"
    hiveContext.sql(creat_sql)
    hiveContext.sql(insert_sql)
  }

  def get_questionnaire_data(hiveContext: HiveContext, yestoday_Date: String) = {
    import org.apache.spark.sql.functions._
    val questionnaire_select_sql: String = "SELECT flymeid, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, count from (SELECT flymeid, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12 AS count FROM \n(SELECT flymeid, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, IF(Q1!=\"\", 1, 0) AS c1, \nIF(Q2!=\"\", 1, 0) AS c2, IF(Q3!=\"\", 1, 0) AS c3, IF(Q4!=\"\", 1, 0) AS c4, IF(Q5!=\"\", 1, 0) AS c5, IF(Q6!=\"\", 1, 0) AS c6, IF(Q7!=\"\", 1, 0) AS c7, IF(Q8!=\"\", 1, 0) AS c8, IF(Q9!=\"\", 1, 0) AS c9, IF(Q10!=\"\", 1, 0) AS c10, IF(Q11!=\"\", 1, 0) AS c11, IF(Q12!=\"\", 1, 0) AS c12 FROM \n(SELECT flymeid, get_json_object(content,'$.0') as Q1, get_json_object(content,'$.1') as Q2, get_json_object(content,'$.2') as Q3, get_json_object(content,'$.3') as Q4, get_json_object(content,'$.4') as Q5, get_json_object(content,'$.5') as Q6, get_json_object(content,'$.6') as Q7, get_json_object(content,'$.7') as Q8, get_json_object(content,'$.8') as Q9, get_json_object(content,'$.9') as Q10, get_json_object(content,'$.10') as Q11, get_json_object(content,'$.11') as Q12, source, posttime, stat_date FROM ext_metis.ods_questionnaire) as t_a\n) as t_b) as t_c GROUP by flymeid, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, count"
    val questionnaire_data = hiveContext.sql(questionnaire_select_sql)
    val refined = questionnaire_data.groupBy("flymeid").agg(max("count") as "max_count").withColumnRenamed("flymeid", "refined_flymeid")
    val questionnaire_data_refined = questionnaire_data.join(refined, questionnaire_data("flymeid") === refined("refined_flymeid") and questionnaire_data("count") === refined("max_count"))
    println("\n\n******************* questionnaire_data: " + questionnaire_data.count() + " ******************")
    println("******************* refined: " + refined.count() + " ******************")
    println("******************* questionnaire_data_refined: " + questionnaire_data_refined.count() + " ****************")

    val select_imei_uid_sql: String = "SELECT imei,uid from  user_profile.edl_device_uid_mz_rel where stat_date=" + yestoday_Date
    val imei_uid = hiveContext.sql(select_imei_uid_sql)
    val questionnaire_refined_df = questionnaire_data_refined.join(imei_uid, questionnaire_data_refined("flymeid") === imei_uid("uid"))

    println("******************* questionnaire_refined_df(key from flymeid to imei): " + questionnaire_refined_df.count() + " ****************\n\n")

    // write DB
    questionnaire_refined_df.registerTempTable("temp")
    val questionnaire_table_name: String = "algo.yf_questionnaire_data"
    val creat_sql: String = "create table if not exists " + questionnaire_table_name + " (flymeid string, Q1 string, Q2 string, Q3 string, Q4 string, Q5 string, Q6 string, Q7 string, Q8 string, Q9 string, Q10 string, Q11 string, Q12 string, count int, refined_flymeid string, max_count int, imei string, uid string) stored as textfile"
    val insert_sql: String = "insert overwrite table " + questionnaire_table_name + " select * from temp"
    hiveContext.sql(creat_sql)
    hiveContext.sql(insert_sql)
  }
}