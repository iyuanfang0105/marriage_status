package MARRIAGE_STATUS

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

object Parent {

  case class Imei_parent(imei: String, parent: String)

  def main(args: Array[String]): Unit = {
    //--- Initialization
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext(sparkConf)
    System.setProperty("user.name", "mzsip")
    System.setProperty("HADOOP_USER_NAME", "mzsip")
    sparkConf.setAppName("YF_ALGO_PARENT_MODEL") //application name
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

    //val today = "20171126"//调试用数据
    val today = args(0)
    val year: Int = today.substring(0,4).trim.toInt
    val month: Int = today.substring(4,6).trim.toInt
    val day: Int = today.substring(6,8).trim.toInt
    val calendar: Calendar = Calendar.getInstance
    calendar.set(year,month-1,day)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)


    val feature_dim: Int = 30000
    val splitChar: String = "\u0001"
    val source_feature_table_name: String = "app_center.adl_fdt_app_adv_model_install"
    val user_age_from_flyme_table_name: String = "algo.yf_user_age_collect_from_flyme"
    val user_behavoir_features_table_name: String = "algo.yf_user_behavior_features_app_install_on_30000_dims"
    val xxx_child_table_name: String = "user_profile.xxx_md5_child"

    val parent_apps_file_name: String = "hdfs://mzcluster/user/mzsip/wangfan/marriage_status/marriaged_apps.txt"
    val parent_apps: RDD[(String, String)] = sc.textFile(parent_apps_file_name).map(v=>{
      val appid_name: Array[String] = v.split(" ")
      val app_id = appid_name(0).trim
      val app_name = appid_name(1).trim
      (app_id, app_name)
    })

    // (imei, appid)
    // val user_app: RDD[(String, String)] = get_app_user(hiveContext, source_feature_table_name, splitChar, yestoday_Date)
    // val parent_labeled_dataset: RDD[(String, Int)]= get_parent_label(user_age_from_flyme_table_name, user_app, parent_apps, hiveContext, yestoday_Date)
    val parent_labeled_dataset_from_xxx_data: RDD[(String, Int)] = get_parent_label_from_xxx_data(xxx_child_table_name, hiveContext, yestoday_Date)

    val balance = false
    // val data_set_0 = get_data_set_for_build_model(parent_labeled_dataset, user_behavoir_features_table_name, feature_dim, balance, hiveContext, yestoday_Date)
    // val result_0 = build_model(data_set_0, 2)
    // println("\n\n ********************** MY MODEL ********************* \n\n")
    // report_model_performance(result_0._2)

    println("\n\n ********************** XXX MODEL ********************* \n\n")
    val data_set_1 = get_data_set_for_build_model(parent_labeled_dataset_from_xxx_data, user_behavoir_features_table_name, feature_dim, balance, hiveContext, yestoday_Date)
    val result_1 = build_model(data_set_1, 2)
    report_model_performance(result_1._2)

    val result_1_all: RDD[(String, String)] = predict(result_1._1, data_set_1).cache()
    val parent_count = result_1_all.filter(_._2 == "parent").count()
    val non_parent_count = result_1_all.filter(_._2 == "non-parent").count()
    println("\n\n********************** parent: " + parent_count + " ****************")
    println("********************** non-parent: " + non_parent_count + " ****************\n\n")
    println("********************** ratio parent/non-parent: " + parent_count*1.0 / non_parent_count + " ****************\n\n")

    val pre_df: DataFrame = result_1_all.repartition(300).map(v => Imei_parent(v._1, v._2)).toDF

    val create_predict_table_name: String = "algo.yf_parent_prediction_base_on_xxx_data"
    println("\n\n ********************* (Strarting)Insert result to yf table *********************\n\n ")
    pre_df.registerTempTable("prediction")
    hiveContext.sql(
     "create table if not exists " +
       create_predict_table_name +
       " (imei string, parent string) partitioned by (stat_date string) stored as textfile")

    hiveContext.sql(
     "insert overwrite table " +
       create_predict_table_name +
       " partition(stat_date = " +
       yestoday_Date +
       " ) select * from prediction")
    println("\n\n ********************* (Done)Insert result to yf table *********************\n\n ")

    // parent_model_validation_with_age(hiveContext)
  }

  def build_model(
                   data_set: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)]),
                   classes_num: Int
                 ): (LogisticRegressionModel, RDD[(String, (Double, Double))]) = {
    println("\n\n ********************* Build Model *************************")
    val trainSet: RDD[(String, LabeledPoint)] = data_set._1 // balance dataset

    val rdd_temp: Array[RDD[(String, LabeledPoint)]] = trainSet.randomSplit(Array(0.8, 0.2))
    val train_rdd: RDD[(String, LabeledPoint)] = rdd_temp(0).cache()
    val valid_rdd: RDD[(String, LabeledPoint)] = rdd_temp(1).cache()
    println("********************* train set number: " + train_rdd.count() + " *************************")
    println("********************* label_0 count: " + train_rdd.filter(_._2.label == 0).count() + " *******************")
    println("********************* label_1 count: " + train_rdd.filter(_._2.label == 1).count() + " ******************* \n\n")
    println("********************* valid set number: " + valid_rdd.count() + " *************************")
    println("********************* label_0 count: " + valid_rdd.filter(_._2.label == 0).count() + " *******************")
    println("********************* label_1 count: " + valid_rdd.filter(_._2.label == 1).count() + " ******************* \n\n")

    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(classes_num).run(train_rdd.map(_._2))
    val valid_result: RDD[(String, (Double, Double))] = valid_rdd.map(v => (v._1, (model.predict(v._2.features), v._2.label)))


    // val multiclassMetrics = new MulticlassMetrics(valid_result.map(_._2))
    // println("\n\n ********************** Precision = " + multiclassMetrics.precision + " ********************* \n\n")
    // println("\n\n ********************** Recall = " + multiclassMetrics.recall + " ********************* \n\n")
    // for (i <- 0 to classes_num-1) {
    //   println("\n\n ********************** Precision_" + i + " = " + multiclassMetrics.precision(i) + " ********************* \n\n")
    // }

    // val binary_class_metrics = new BinaryClassificationMetrics(valid_result.map(_._2))
    // val roc = binary_class_metrics.roc()
    // val au_roc = binary_class_metrics.areaUnderROC()
    // println("\n\n ********************** AUROC: " + au_roc + " ********************* \n\n")

    return (model, valid_result)
  }

  def predict(model: LogisticRegressionModel, dataset: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)])): RDD[(String, String)] = {
    val prediction: RDD[(String, String)] = dataset._2.map(v => {
      if (model.predict(v._2.features) == 1d)
        (v._1, "parent")
      else
        (v._1, "non-parent")
    })
    val pre: RDD[(String, String)] = dataset._1.map(v => (v._1, if (v._2.label == 1d) "parent" else "non-parent")).union(prediction)
    return pre
  }

  def report_model_performance(valid_result: RDD[(String, (Double, Double))]): Unit ={
    val roc = ROC(valid_result.map(_._2))

    println("\n\n ********************** AUROC: " + roc._1 + " *********************")
    println("********************** Accuracy: " + roc._2 + " *********************")

    // val cross_validate_result = my_valid_result.join(xxx_valid_result).filter(v => v._2._1._2 == v._2._2._2)
    // val cross_valid_result_my = ROC(cross_validate_result.map(_._2._1))
    // val cross_valid_result_xxx = ROC(cross_validate_result.map(_._2._2))
    // println("\n\n The count of cross validation result: " + cross_validate_result.count() + " ****************** \n\n")
    // println("\n\n The count of cross_valid__result_my: " + cross_validate_result.map(_._2._1).count() + " ****************** \n\n")
    // println("\n\n The count of cross_valid__result_xxx: " + cross_validate_result.map(_._2._2).count() + " ****************** \n\n")
    // println("\n\n ********************** Cross AUROC of my: " + cross_valid_result_my + " ********************* \n\n")
    // println("\n\n ********************** Cross AUROC of xxx: " + cross_valid_result_xxx + " ********************* \n\n")
  }

  def ROC(valid_result: RDD[(Double, Double)]): (Double, Double) = {
    val binary_class_metrics = new BinaryClassificationMetrics(valid_result)
    val roc = binary_class_metrics.roc()
    val au_roc = binary_class_metrics.areaUnderROC()
    val accuracy = valid_result.filter(v => v._1 == v._2).count() * 1.0 / valid_result.count()
    // println("\n\n ********************** AUROC: " + au_roc + " ********************* \n\n")
    return (au_roc, accuracy)
  }

  def get_data_set_for_build_model(
                                    parent_labeled_dataset: RDD[(String, Int)],
                                    user_behavoir_features_table_name: String,
                                    feature_dim: Int,
                                    balance: Boolean,
                                    hiveContext: HiveContext,
                                    yestoday_Date: String): (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)]) = {
    // get  latest date of user behavior features
    // val select_latest_date_sql = "SELECT stat_date from " + user_behavoir_features_table_name + " GROUP by stat_date ORDER by stat_date DESC"
    // val latest_date: String = hiveContext.sql(select_latest_date_sql).first()(0).toString
    val select_latest_date_sql = "show PARTITIONS " + user_behavoir_features_table_name
    val latest_date: String = hiveContext.sql(select_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("\n\n ***************** get_data_set_for_build_model  ************* ")
    println("***************** The latest date of user behavior: " + latest_date.toString() + " *************")
    // user behavior features
    val select_imei_feature_sql: String = "select * from " + user_behavoir_features_table_name + " where stat_date=" + latest_date
    val imei_feature_df: DataFrame = hiveContext.sql(select_imei_feature_sql)

    val imei_feature_rdd: RDD[(String, String)] = imei_feature_df.rdd.map(v => (v(0).toString, v(1).toString))
    println("********************* The number of user hehavior features data: " + imei_feature_rdd.count() + " ***********************")

    // building the train and predict set
    val predict_set_rdd: RDD[(String, LabeledPoint)] = imei_feature_rdd.subtractByKey(parent_labeled_dataset).map(v => {
      val imei: String = v._1
      val feature_str: String = v._2
      val label: Int = -1
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
      (imei, label, index_array.toArray, value_array.toArray)
    }).filter(v => v._3.length > 0).map(v => (v._1, new LabeledPoint(v._2, Vectors.sparse(feature_dim, v._3, v._4))))
    println("********************* The size of predict set: " + predict_set_rdd.count() + " ***********************")

    val train_set_rdd: RDD[(String, LabeledPoint)] = imei_feature_rdd.join(parent_labeled_dataset).map(v => {
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
    println("********************* The size of train set: " + train_set_rdd.count() + " *********************** \n\n")

    if (balance) {
      return (balance_dataset(train_set_rdd), predict_set_rdd)
    }else{
      return (train_set_rdd, predict_set_rdd)
    }
  }

  def balance_dataset(data_set: RDD[(String, LabeledPoint)]): RDD[(String, LabeledPoint)] = {
    println("\n\n ******************************* Blance dataset ************************************ ")
    val label_0 = data_set.filter(v => v._2.label == 0)
    val label_0_count = label_0.count()
    val label_1 = data_set.filter(v => v._2.label == 1)
    val label_1_count = label_1.count()
    println("******************************* label_0: " + label_0_count + " ************************************")
    println("******************************* label_1: " + label_1_count + " ************************************ \n\n")

    val sample_num = math.min(label_0_count, label_1_count)
    val seed = 1234
    return label_0.sample(false, sample_num * 1.0 / label_0_count, seed).union(label_1.sample(false, sample_num * 1.0 / label_1_count, seed))
  }

  def get_parent_label(user_age_from_flyme_table_name: String,
                       user_app: RDD[(String, String)], // (imei, app_id)
                       parent_apps: RDD[(String, String)], // (app_id, app_name)
                       hiveContext: HiveContext,
                       yestoday_Date: String): RDD[(String, Int)] = {
    // (imei, age)
    val parent_candidates_by_age: (RDD[(String, Int)], RDD[(String, Int)], RDD[(String, Int)]) = get_parent_candidates_by_age(user_age_from_flyme_table_name, hiveContext, yestoday_Date)
    val user_older_than_35: RDD[(String, Int)] = parent_candidates_by_age._1
    val user_between_18_and_35: RDD[(String, Int)] = parent_candidates_by_age._2
    val user_smaller_than_18: RDD[(String, Int)] = parent_candidates_by_age._3

    // (imei, appid)
    val parent_labeled_by_parent_app: RDD[(String, String)] = user_app.map(v => (v._2, v._1)).join(parent_apps).map(v => (v._2._1, v._1)).distinct()
    println("\n\n **********　The number of parent labeled by parent apps: " + parent_labeled_by_parent_app.count() + " ************ \n\n")

    // label the user between 18 and 35 according app
    val user_between_18_and_35_parent_label_1: RDD[(String, Int)] = user_between_18_and_35.join(parent_labeled_by_parent_app).map(v => (v._1, 1)).distinct()
    val user_between_18_and_35_parent_label_0: RDD[(String, Int)] = user_between_18_and_35.subtractByKey(parent_labeled_by_parent_app).map(v => (v._1, 0)).distinct()
    println("\n\n ********** The number of parent label 1 between 18 and 35: " + user_between_18_and_35_parent_label_1.count() + " ************ \n\n")
    println("\n\n ********** The number of parent label 0 between 18 and 35: " + user_between_18_and_35_parent_label_0.count() + " ************ \n\n")

    // (imei, 1)
    val parent_label_1: RDD[(String, Int)] = user_older_than_35.map(v => (v._1, 1)).union(user_between_18_and_35_parent_label_1).distinct()
    val parent_label_0: RDD[(String, Int)] = user_smaller_than_18.map(v => (v._1, 1)).union(user_between_18_and_35_parent_label_0).distinct()
    println("\n\n ********** The number of parent label 1: " + parent_label_1.count() + " ************ \n\n")
    println("\n\n ********** The number of parent label 0: " + parent_label_0.count() + " ************ \n\n")

    return parent_label_0.union(parent_label_1)
  }

  def get_parent_label_from_xxx_data (
                                       xxx_child_table_name: String,
                                       hiveContext: HiveContext,
                                       yestoday_Date: String
                                     ): RDD[(String, Int)] = {
    val xxx_child_data_select_sql: String = "select * from " + xxx_child_table_name + " t_a left join user_profile.xxx_md5 t_b on lower(t_a.imeimd5)=lower(t_b.imeimd5) where t_b.imei is not null"
    val xxx_child_data_df: DataFrame = hiveContext.sql(xxx_child_data_select_sql).select("imei", "child")
    val xxx_child_data_rdd: RDD[(String, Int)] = xxx_child_data_df.rdd.map(v => (v.get(0).toString, v.get(1).toString.toInt))
    print("\n\n ************* The number of xxx_child_data: " + xxx_child_data_rdd.count() + " ****************** \n\n")
    return xxx_child_data_rdd
  }

  // get the parent candidates according age (marriaged: older than 35 and unmarriaged: smaller than 35)
  def get_parent_candidates_by_age(
                        user_age_from_flyme_table_name: String,
                        hiveContext: HiveContext,
                        yestoday_Date: String
                      ): (RDD[(String, Int)], RDD[(String, Int)], RDD[(String, Int)]) = {
    // val select_latest_date_sql = "SELECT stat_date from " + user_age_from_flyme_table_name + " GROUP by stat_date ORDER by stat_date DESC"
    // val latest_date: String = hiveContext.sql(select_latest_date_sql).first()(0).toString
    val select_latest_date_sql = "show PARTITIONS " + user_age_from_flyme_table_name
    val latest_date: String = hiveContext.sql(select_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("\n\n ***************** The latest date of user_age_from_flyme: " + latest_date + " ************* \n\n")
    val select_user_age_sql: String = "select imei, age from " + user_age_from_flyme_table_name + " where stat_date=" + latest_date
    // age below 35 should be candidates to be parents
    val user_age: RDD[(String, Int)] = hiveContext.sql(select_user_age_sql).rdd.map(v => (v(0).toString, v(1).toString.toInt))
    val user_older_than_35: RDD[(String, Int)] = user_age.filter(v => v._2 > 35).distinct()
    val user_between_18_and_35: RDD[(String, Int)] = user_age.filter(v => (v._2 <= 35 && v._2 >=18)).distinct()
    val user_smaller_than_18: RDD[(String, Int)] = user_age.filter(v => v._2 < 18).distinct()

    val user_teen: RDD[(String, Int)] = user_age.filter(v => v._2 < 18)
    println("\n\n ***************** The number of candidates(older than 35): " + user_older_than_35.count() + " ********** \n\n")
    println("\n\n ***************** The number of candidates(between 18 and 35): " + user_between_18_and_35.count() + " ********** \n\n")
    println("\n\n ***************** The number of candidates(smaller than 18): " + user_smaller_than_18.count() + " ********** \n\n")

    return (user_older_than_35, user_between_18_and_35, user_smaller_than_18)
  }

  // get the user behavior of install app, format: (imei, appid)
  def get_app_user(
                    hiveContext: HiveContext,
                    source_feature_table_name: String,
                    splitChar: String,
                    yestoday_Date: String): RDD[(String, String)] = {
    val select_latest_date_sql = "show PARTITIONS " + source_feature_table_name
    val latest_date: String = hiveContext.sql(select_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("\n\n ***************** The latest date of user_age_from_flyme: " + latest_date + " ************* \n\n")

    val select_source_feature_table_sql: String = "select * from " + source_feature_table_name + " where stat_date = " + latest_date
    //(imei, feature)
    val imei_features_df: DataFrame = hiveContext.sql(select_source_feature_table_sql)
    //println("count of imei_features_df for " + sqls_dataType(i)._2 + ": " + imei_features_df.count())
    val imei_features_rdd1: RDD[(String, Array[String])] = imei_features_df.map(v => (v(0).toString, v(1).toString.trim.split(" ")))
    val imei_features_rdd2 = imei_features_rdd1.filter(_._2.length > 0)
    //println("count of imei_features_rdd2 for " + sqls_dataType(i)._2 + ": " + imei_features_rdd2.count)
    val imei_features_rdd3 = imei_features_rdd2.mapPartitions(iter => {
      new Iterator[(String, String)]() {
        var count: Int = 0
        var value: (String, Array[String]) = iter.next()
        override def hasNext: Boolean = {
          if (count < value._2.length)
            true
          else {
            if (iter.hasNext) {
              count = 0
              value = iter.next()
              true
            }
            else
              false
          }
        }

        override def next(): (String, String) = {
          count += 1
          (value._2(count - 1), value._1)
        }
      }
    })
    //println("count of imei_features_rdd3 for " + sqls_dataType(i)._2 + ": " +  + imei_features_rdd3.count())
    val imei_features_rdd4 = imei_features_rdd3.filter(_._1.trim.split(":").length == 2).map(v => {
      val array: Array[String] = v._1.trim.split(":")
      (v._2, array(0).trim.substring(2))
    })
    //println("count of imei_features_rdd4 for" + sqls_dataType(i)._2 + ": " + + imei_features_rdd4.count())
    imei_features_rdd4
  }

  def parent_model_validation_with_age(hiveContext: HiveContext) ={
    // lx_age_appuse_prediction yf_parent_prediction_base_on_xxx_data
    val parent_table: String = "algo.yf_parent_prediction_base_on_xxx_data"
    val age_table: String = "algo.lx_age_appuse_prediction"
    val select_parent_latest_date_sql = "show PARTITIONS " + parent_table
    val parent_latest_date: String = hiveContext.sql(select_parent_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("\n\n ***************** The latest date of parent table: " + parent_latest_date + " ************* ")

    val select_age_latest_date_sql = "show PARTITIONS " + age_table
    val age_latest_date: String = hiveContext.sql(select_age_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("***************** The latest date of age table: " + age_latest_date + " ************* ")

    val parent_data_select_sql: String = "select * from " + parent_table + " where stat_date=" + parent_latest_date
    val age_data_select_sql: String = "select * from " + age_table + " where stat_date=" + age_latest_date

    val parent_data_df: DataFrame = hiveContext.sql(parent_data_select_sql)
    val age_data_df: DataFrame = hiveContext.sql(age_data_select_sql)
    // imei, parent, age
    val parent_age = parent_data_df.join(age_data_df, parent_data_df("imei") === age_data_df("imei")).rdd.map(v => (v(0).toString, v(1).toString, v(4).toString.toDouble))

    val parent_case = parent_age.filter(v => v._2 == "parent")
    val non_parent_case = parent_age.filter(v => v._2 == "non-parent")

    val parent_badcase_0 = parent_age.filter(v => v._2 == "parent" && v._3 == 0d)
    val parent_badcase_1 = parent_age.filter(v => v._2 == "parent" && v._3 == 1d)

    val non_parent_badcase_0 = parent_age.filter(v => v._2 == "non-parent" && v._3 == 3)
    val non_parent_badcase_1 = parent_age.filter(v => v._2 == "non-parent" && v._3 == 4)

    println("***************** parent count: " + parent_case.count() + " ************* ")
    println("***************** non-parent count: " + non_parent_case.count() + " ************* ")

    println("***************** parent　badcase (7~14): " + parent_badcase_0.count() + " ************* ")
    println("***************** ratio (parent　badcase (7~14)): " + parent_badcase_0.count() * 1.0 / parent_case.count() + " ************* ")
    println("***************** parent　badcase (15~22): " + parent_badcase_1.count() + " ************* ")
    println("***************** ratio (parent　badcase (15~22)): " + parent_badcase_1.count() * 1.0 / parent_case.count() + " ************* \n\n")

    println("***************** parent　badcase (36~45): " + non_parent_badcase_0.count() + " ************* ")
    println("***************** ratio (parent　badcase (36~45)): " + non_parent_badcase_0.count() * 1.0 / non_parent_case.count() + " ************* ")
    println("***************** parent　badcase (46~76): " + non_parent_badcase_1.count() + " ************* ")
    println("***************** ratio (parent　badcase (46~76)): " + non_parent_badcase_1.count() * 1.0 / non_parent_case.count() + " ************* \n\n")

  }
}
