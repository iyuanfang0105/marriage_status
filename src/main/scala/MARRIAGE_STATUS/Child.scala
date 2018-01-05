package MARRIAGE_STATUS

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext

import scala.collection.mutable.ArrayBuffer

object Child {

  def main(args: Array[String]): Unit = {
    //--- Initialization
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext(sparkConf)
    System.setProperty("user.name", "mzsip")
    System.setProperty("HADOOP_USER_NAME", "mzsip")
    sparkConf.setAppName("YF_ALGO_CHILD_MODEL") //application name
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

    val today = "20171226"
    //调试用数据
    // val today = args(0)
    val year: Int = today.substring(0, 4).trim.toInt
    val month: Int = today.substring(4, 6).trim.toInt
    val day: Int = today.substring(6, 8).trim.toInt
    val calendar: Calendar = Calendar.getInstance
    calendar.set(year, month - 1, day)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)
    println("\n\n******************* yestoday_Date: " + yestoday_Date + " ********************\n\n")

    val feature_dim: Int = 30000
    val user_age_from_flyme_table_name: String = "algo.yf_user_age_collect_from_flyme"
    val user_behavoir_features_table_name: String = "algo.yf_user_behavior_features_app_install_on_30000_dims"
    val xxx_child_table_name: String = "user_profile.xxx_md5_child"
    val child_model_predictions_table_name: String = "algo.yf_parent_prediction_base_on_xxx_data"
    val xxx_child_stage_table_name: String = "algo.yf_xxx_labels"

    val parent_labeled_dataset_from_xxx_data: RDD[(String, Int)] = get_parent_label_from_xxx_data(xxx_child_table_name, hiveContext, yestoday_Date)
    val parent_labeled_dataset_from_model_predictions: RDD[(String, Int)] = get_parent_label_from_model_predictions(child_model_predictions_table_name, hiveContext, yestoday_Date)
    val child_stage_dataset_from_xxx_data: RDD[(String, Int)] = get_child_stage_label_from_xxx_data(xxx_child_stage_table_name, hiveContext, yestoday_Date)
    val child_stage_dataset_refined_by_parent_predictions = child_stage_dataset_from_xxx_data.join(parent_labeled_dataset_from_model_predictions.filter(_._2 == 1)).map(v => (v._1, v._2._1))
    val child_stage_dataset_confilict_with_parent_predictions = child_stage_dataset_from_xxx_data.join(parent_labeled_dataset_from_model_predictions.filter(_._2 == 0)).map(v => (v._1, v._2._1))
    val child_stage_dataset_refined_by_xxx_parent = child_stage_dataset_from_xxx_data.join(parent_labeled_dataset_from_xxx_data.filter(_._2 == 1)).map(v => (v._1, v._2._1))
    val child_stage_dataset_confilict_with_xxx_parent = child_stage_dataset_from_xxx_data.join(parent_labeled_dataset_from_xxx_data.filter(_._2 == 0)).map(v => (v._1, v._2._1))


    val child_stage_refined_by_parent_prediction_count = child_stage_dataset_refined_by_parent_predictions.count()
    val child_stage_conflict_with_parent_prediction_count = child_stage_dataset_confilict_with_parent_predictions.count()
    val child_stage_refined_by_parent_prediction_baby_count = child_stage_dataset_refined_by_parent_predictions.filter(_._2 == 1).count()
    val child_stage_refined_by_parent_prediction_pregnent_count = child_stage_dataset_refined_by_parent_predictions.filter(_._2 == 2).count()
    val child_stage_refined_by_parent_prediction_teen_count = child_stage_dataset_refined_by_parent_predictions.filter(_._2 == 3).count()

    val child_stage_refined_by_xxx_parent_count = child_stage_dataset_refined_by_xxx_parent.count()
    val child_stage_conflict_with_xxx_parent_count = child_stage_dataset_confilict_with_xxx_parent.count()
    val child_stage_refined_by_xxx_parent_baby_count = child_stage_dataset_refined_by_xxx_parent.filter(_._2 == 1).count()
    val child_stage_refined_by_xxx_parent_pregnent_count = child_stage_dataset_refined_by_xxx_parent.filter(_._2 == 2).count()
    val child_stage_refined_by_xxx_parent_teen_count = child_stage_dataset_refined_by_xxx_parent.filter(_._2 == 3).count()

    println("\n\n******************* child_stage_refined_by_parent_prediction_count: " + child_stage_refined_by_parent_prediction_count + " ***************")
    println("******************* child_stage_conflict_with_parent_prediction_count: " + child_stage_conflict_with_parent_prediction_count + " ***************")
    println("********************* child_stage_refined_by_parent_prediction_baby_count: " + child_stage_refined_by_parent_prediction_baby_count + " ratio: " + child_stage_refined_by_parent_prediction_baby_count*1.0 / child_stage_refined_by_parent_prediction_count + " *****************")
    println("********************* child_stage_refined_by_parent_prediction_pregnent_count: " + child_stage_refined_by_parent_prediction_pregnent_count + " ratio: " + child_stage_refined_by_parent_prediction_pregnent_count*1.0 / child_stage_refined_by_parent_prediction_count + " *****************")
    println("********************* child_stage_refined_by_parent_prediction_teen_count: " + child_stage_refined_by_parent_prediction_teen_count + " ratio: " + child_stage_refined_by_parent_prediction_teen_count*1.0 / child_stage_refined_by_parent_prediction_count + " *****************\n\n")

    println("********************* child_stage_refined_by_xxx_parent_count: " + child_stage_refined_by_xxx_parent_count + " ***************")
    println("********************* child_stage_conflict_with_xxx_parent_count: " + child_stage_conflict_with_xxx_parent_count + " ***************")
    println("********************* child_stage_refined_by_xxx_parent_baby_count: " + child_stage_refined_by_xxx_parent_baby_count + " ratio: " + child_stage_refined_by_xxx_parent_baby_count*1.0 / child_stage_refined_by_xxx_parent_count + " *****************")
    println("********************* child_stage_refined_by_xxx_parent_pregnent_count: " + child_stage_refined_by_xxx_parent_pregnent_count + " ratio: " + child_stage_refined_by_xxx_parent_pregnent_count*1.0 / child_stage_refined_by_xxx_parent_count + " *****************")
    println("********************* child_stage_refined_by_xxx_parent_teen_count: " + child_stage_refined_by_xxx_parent_teen_count + " ratio: " + child_stage_refined_by_xxx_parent_teen_count*1.0 / child_stage_refined_by_xxx_parent_count + " *****************\n\n")

  }

  def get_parent_label_from_xxx_data(xxx_child_table_name: String,
                                     hiveContext: HiveContext,
                                     yestoday_Date: String
                                    ): RDD[(String, Int)] = {
    val xxx_child_data_select_sql: String = "select * from " + xxx_child_table_name + " t_a left join user_profile.xxx_md5 t_b on lower(t_a.imeimd5)=lower(t_b.imeimd5) where t_b.imei is not null"
    val xxx_child_data_df: DataFrame = hiveContext.sql(xxx_child_data_select_sql).select("imei", "child")
    val xxx_child_data_rdd: RDD[(String, Int)] = xxx_child_data_df.rdd.map(v => (v.get(0).toString, v.get(1).toString.toInt))

    val xxx_child_data_count = xxx_child_data_rdd.count()
    val xxx_parent_count = xxx_child_data_rdd.filter(_._2 == 1).count()
    val xxx_non_parent_count = xxx_child_data_rdd.filter(_._2 == 0).count()
    println("\n\n ***************** xxx_child_data_count: " + xxx_child_data_count + " ******************")
    println("***************** xxx_parent_count: " + xxx_parent_count + " ratio: " + xxx_parent_count*1.0 / xxx_child_data_count + " ******************")
    println("***************** xxx_non_parent_count: " + xxx_non_parent_count + " ratio: " + xxx_non_parent_count*1.0 / xxx_child_data_count + " ******************")

    return xxx_child_data_rdd
  }

  def get_parent_label_from_model_predictions(child_model_predictions_table_name: String,
                                              hiveContext: HiveContext,
                                              yestoday_Date: String): RDD[(String, Int)] = {
    val child_model_predictions_data_select_sql: String = "select imei, parent from " + child_model_predictions_table_name + " where stat_date=" + yestoday_Date
    val child_model_predictions_data: RDD[(String, Int)] = hiveContext.sql(child_model_predictions_data_select_sql).rdd.map(v => {
      if(v(1).toString == "parent")
        (v(0).toString, 1)
      else
        (v(0).toString, 0)
    })
    val child_model_predictions_data_count = child_model_predictions_data.count()
    val child_model_predictions_parent_count = child_model_predictions_data.filter(_._2 == 1).count()
    val child_model_predictions_non_parent_count = child_model_predictions_data.filter(_._2 == 0).count()
    println("\n\n**************** child_model_predictions_data_count: " + child_model_predictions_data_count + " ******************")
    println("**************** child_model_predictions_parent_count: " + child_model_predictions_parent_count + " ratio: " + child_model_predictions_parent_count*1.0 / child_model_predictions_data_count + " **************")
    println("**************** child_model_predictions_non_parent_count: " + child_model_predictions_non_parent_count + " ratio: " + child_model_predictions_non_parent_count*1.0 / child_model_predictions_data_count + " **************\n\n")
    return child_model_predictions_data
  }

  def get_child_stage_label_from_xxx_data(xxx_child_stage_table_name: String,
                                          hiveContext: HiveContext,
                                          yestoday_Date: String): RDD[(String, Int)] = {
    val xxx_child_stage_sql: String = "select imei, child_stage from " + xxx_child_stage_table_name + " where child_stage=\"婴幼儿\" or child_stage=\"孕育期\" or child_stage=\"青少年\""
    val child_stage_labeled_data = hiveContext.sql(xxx_child_stage_sql).rdd.map(v => {
      var label: Int = 0
      if(v(1).toString == "婴幼儿")
        label = 1
      else if(v(1).toString == "孕育期")
        label = 2
      else if(v(1).toString == "青少年")
        label = 3
      (v(0).toString, label)
    }).filter(_._2 != 0)

    val child_stage_labeled_data_count = child_stage_labeled_data.count()
    val baby_data_count = child_stage_labeled_data.filter(_._2 == 1).count()
    val pregnent_data_count = child_stage_labeled_data.filter(_._2 == 2).count()
    val teen_data_count = child_stage_labeled_data.filter(_._2 == 3).count()

    println("\n\n******************* child_stage_labeled_data_count: " + child_stage_labeled_data_count + " ***************")
    println("********************* baby_data_count: " + baby_data_count + " ratio: " + baby_data_count*1.0 / child_stage_labeled_data_count + " *****************")
    println("********************* pregnent_data_count: " + pregnent_data_count + " ratio: " + pregnent_data_count*1.0 / child_stage_labeled_data_count + " *****************")
    println("********************* teen_data_count: " + teen_data_count + " ratio: " + teen_data_count*1.0 / child_stage_labeled_data_count + " *****************\n\n")

    return child_stage_labeled_data
  }

  def get_child_stage_label_from_questionnaire(hiveContext: HiveContext,
                                               yestoday_Date: String): RDD[(String, Double)] = {
    val questionnaire_select_sql: String = "SELECT imei, Q7 from (SELECT t_a.flymeid, t_a.Q7 from (SELECT flymeid, get_json_object(content,'$.0') as Q1, get_json_object(content,'$.1') as Q2, get_json_object(content,'$.2') as Q3, get_json_object(content,'$.3') as Q4, get_json_object(content,'$.4') as Q5, get_json_object(content,'$.5') as Q6, get_json_object(content,'$.6') as Q7, get_json_object(content,'$.7') as Q8, get_json_object(content,'$.8') as Q9, get_json_object(content,'$.9') as Q10, get_json_object(content,'$.10') as Q11, get_json_object(content,'$.11') as Q12, source, posttime, stat_date FROM ext_metis.ods_questionnaire) as t_a where t_a.Q1!=\"\" and t_a.Q2!=\"\"and t_a.Q3!=\"\" and t_a.Q4!=\"\"and t_a.Q5!=\"\" and t_a.Q6!=\"\" and t_a.Q7!=\"\"and t_a.Q8!=\"\" and t_a.Q9!=\"\"and t_a.Q10!=\"\" and t_a.Q11!=\"\" and t_a.Q12!=\"\") as t_b join (SELECT imei,uid from  user_profile.edl_device_uid_mz_rel where stat_date=" + yestoday_Date + ") as t_c where t_b.flymeid=t_c.uid"
    val questionnaire_df = hiveContext.sql(questionnaire_select_sql)
    val questionnaire_refined = questionnaire_df.rdd.filter(v => v(1).toString.length == 1).map(v => (v(0).toString, v(1).toString))
    val questionnaire_count = questionnaire_df.count()
    val questionnaire_refined_count = questionnaire_refined.count()
    println("\n\n********************** questionnaire_count: " + questionnaire_count + " ********************")
    println("********************** questionnaire_refined_count: " + questionnaire_refined_count + " ********************\n\n")

    val questionnaire_refined_summary = questionnaire_refined.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    questionnaire_refined_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))

    val questionnaire_refined_label = questionnaire_refined.map(v => {
      var label: Double = 0
      if(v._2 == "A")
        label = 1
      else if(v._2 == "B")
        label = 2
      else if(v._2 == "C")
        label = 3
      else if(v._2 == "D")
        label = 4
      else if(v._2 == "E")
        label = 5
      else if(v._2 == "F")
        label = 6
      else if(v._2 == "H")
        label = 7
      else label = 8
      (v._1, label)
    })
    val questionnaire_refined_label_summary = questionnaire_refined_label.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    questionnaire_refined_label_summary.sortByKey().collect().foreach(v => println(v._1 + " " + v._2.toString))

    return questionnaire_refined_label.filter(_._2 !=0).filter(_._2 != 8)
  }
}