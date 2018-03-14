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

object Marriage_Parent_Child {

  case class Imei_MPC(imei: String, marriage: String, parent: String, child_stage: String)

  def main(args: Array[String]): Unit = {
    //--- Initialization
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext(sparkConf)
    System.setProperty("user.name", "mzsip")
    System.setProperty("HADOOP_USER_NAME", "mzsip")
    sparkConf.setAppName("YF_ALGO_MARRIAGE_PARENT_CHILD_MODEL") //application name
    //---
    sc.hadoopConfiguration.set("mapred.output.compress", "false")

    Logger.getRootLogger.setLevel(Level.ERROR)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)


    val hiveContext: HiveContext = new HiveContext(sc)
    hiveContext.setConf("mapred.output.compress", "false")
    hiveContext.setConf("hive.exec.compress.output", "false")
    hiveContext.setConf("mapreduce.output.fileoutputformat.compress", "false")
    import hiveContext.implicits._

    // val today = "20171225"
    //调试用数据
    val today = args(0)
    val year: Int = today.substring(0, 4).trim.toInt
    val month: Int = today.substring(4, 6).trim.toInt
    val day: Int = today.substring(6, 8).trim.toInt
    val calendar: Calendar = Calendar.getInstance
    calendar.set(year, month - 1, day)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)
    println("\n\n******************* yestoday_Date: " + yestoday_Date + " ********************\n\n")

    val feature_dim: Int = 30000
    val balance = false
    val user_behavoir_features_table_name: String = "algo.yf_user_behavior_features_app_install_on_30000_dims"
    val questionnaire_table_name: String = "algo.yf_questionnaire_data"
    val predictions_table_name: String = "algo.yf_marriage_parent_child_stage"

    // marriage model based on questionnaire
    println("\n\n ************** marriage model based on questionnaire **************** \n\n")
    val marriage_class_num = 2
    val marriage_labeled_dataseet_from_questionnaire = get_marriage_label_from_questionnaire(hiveContext, questionnaire_table_name)
    val data_set_marriage_questionnaire = get_data_set_for_build_model(marriage_labeled_dataseet_from_questionnaire, user_behavoir_features_table_name, feature_dim, balance, hiveContext, yestoday_Date)
    val result_marriage_questionnaire = build_model(data_set_marriage_questionnaire, marriage_class_num)
    val marriage_questionnaire_roc = ROC(result_marriage_questionnaire._2.map(_._2))
    println("\n\n*************** ROC_marriage_questionnaire_model: " + marriage_questionnaire_roc._1 + " ACCU_marriage_questionnaire_model: " + marriage_questionnaire_roc._2 + " **************\n\n")
    val pred_marraige = predict_marriage(result_marriage_questionnaire._1, data_set_marriage_questionnaire)

    // parent model based on questionnaire data
    println("\n\n ************** parent model based on questionnaire data ****************")
    val parent_class_num = 2
    val parent_labeled_dataset_from_questionnaire_data: RDD[(String, Int)] = get_parent_label_from_questionnaire(hiveContext, questionnaire_table_name)
    val data_set_parent_questionnaire = get_data_set_for_build_model(parent_labeled_dataset_from_questionnaire_data, user_behavoir_features_table_name, feature_dim, balance, hiveContext, yestoday_Date)
    val result_parent_model_questionnaire = build_model(data_set_parent_questionnaire, parent_class_num)
    val parent_model_roc_questionnaire = ROC(result_parent_model_questionnaire._2.map(_._2))
    println("*************** ROC_parent_model_questionnaire: " + parent_model_roc_questionnaire._1 + " ACCU_parent_model_questionnaire: " + parent_model_roc_questionnaire._2 + " ***************\n\n")
    val pred_parent = predict_parent(result_parent_model_questionnaire._1, data_set_parent_questionnaire)

    // child model based on xxx data
    println("\n\n ************** child model based on xxx data ****************")
    val child_class_num_xxx = 3
    val child_labeled_dataset_from_xxx: RDD[(String, Int)] = get_child_stage_label_from_xxx_data(hiveContext, yestoday_Date)
    val data_set_child_xxx = get_data_set_for_build_model(child_labeled_dataset_from_xxx, user_behavoir_features_table_name, feature_dim, balance, hiveContext, yestoday_Date)
    val result_child_model_xxx = build_model(data_set_child_xxx, child_class_num_xxx)
    val child_model_confusion_matrix_xxx = confusion_matrix(result_child_model_xxx._2.map(_._2))
    println("******************* Precision_child_model: " + child_model_confusion_matrix_xxx.precision + " Recall_child_model: " + child_model_confusion_matrix_xxx.recall + " ***************")
    for(i <- 0 until child_class_num_xxx){
      println("*************** Precision_" + i.toString + ": " + child_model_confusion_matrix_xxx.precision(i) + " Recall_" + i.toString +": " + child_model_confusion_matrix_xxx.recall(i)+ " ***************")
    }
    val pred_child_stage = predict_child_stage(result_child_model_xxx._1, data_set_child_xxx)

    // merged result and insert to db
    val merged_pred = merge_predictions(pred_marraige, pred_parent, pred_child_stage)
    save_to_db(hiveContext, merged_pred, predictions_table_name, yestoday_Date)
  }

  def get_marriage_label_from_flyme_age(hiveContext: HiveContext, yestoday_Date: String): RDD[(String, Int)] = {
    // val select_latest_date_sql = "SELECT stat_date from " + user_age_from_flyme_table_name + " GROUP by stat_date ORDER by stat_date DESC"
    // val latest_date: String = hiveContext.sql(select_latest_date_sql).first()(0).toString
    // println("\n\n ***************** The latest date of user_age_from_flyme: " + latest_date + " ************* \n\n")
    val user_age_from_flyme_table_name: String = "algo.yf_user_age_collect_from_flyme"
    val select_user_age_sql: String = "select * from " + user_age_from_flyme_table_name + " where stat_date=" + yestoday_Date
    val user_marriage_label: RDD[(String, Int)] = hiveContext.sql(select_user_age_sql).rdd.map(v => {
      val imei: String = v(0).toString
      val age: Int = v(2).toString.toInt
      var label: Int = 0
      if (age <= 35)
        label = 1
      if (age > 35)
        label = 2
      (imei, label)
    }).filter(_._2 != 0).reduceByKey((a, b) => a)
    val user_marriage_label_summary = user_marriage_label.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    user_marriage_label_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    return user_marriage_label.map(v => (v._1, v._2-1))
  }

  def get_marriage_label_from_questionnaire(hiveContext: HiveContext, questionnaire_table_name: String): RDD[(String, Int)] = {
    val select_sql: String = "SELECT imei, q4 from " + questionnaire_table_name + " where q4==\"A\" OR q4==\"B\" OR q4==\"C\" OR q4==\"D\""
    val user_marriage_label_raw = hiveContext.sql(select_sql).rdd.map(v => (v(0).toString, v(1).toString))
    val user_marriage_label_raw_summary = user_marriage_label_raw.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    user_marriage_label_raw_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    val user_marriage_label: RDD[(String, Int)] = user_marriage_label_raw.map(v => {
      var label: Int = 0
      if (v._2 == "A")
        label = 1
      else if (v._2 == "B")
        label = 2
      else label = 3
      (v._1, label)
    }).filter(_._2 != 0).filter(_._2 != 3).reduceByKey((a, b) => a)
    val user_marriage_label_summary = user_marriage_label.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    user_marriage_label_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    return user_marriage_label.map(v => (v._1, v._2-1))
  }

  def get_parent_label_from_xxx_data(hiveContext: HiveContext): RDD[(String, Int)] = {
    val xxx_parent_table_name: String = "user_profile.xxx_md5_child"
    val xxx_parent_data_select_sql: String = "select * from " + xxx_parent_table_name + " t_a left join user_profile.xxx_md5 t_b on lower(t_a.imeimd5)=lower(t_b.imeimd5) where t_b.imei is not null"
    val xxx_parent_data_df: DataFrame = hiveContext.sql(xxx_parent_data_select_sql).select("imei", "child")
    val xxx_parent_data_rdd: RDD[(String, Int)] = xxx_parent_data_df.rdd.map(v => (v.get(0).toString, v.get(1).toString.toInt)).reduceByKey((a, b) => a)
    print("\n\n************* The number of xxx_parent_data: " + xxx_parent_data_rdd.count() + " ****************** \n\n")
    return xxx_parent_data_rdd
  }

  def get_parent_label_from_questionnaire(hiveContext: HiveContext, questionnaire_table_name: String): RDD[(String, Int)] = {
    val questionnaire_select_sql: String = "SELECT imei, Q7 from " + questionnaire_table_name + " where Q7!=\"\""
    val questionnaire_df = hiveContext.sql(questionnaire_select_sql)
    val questionnaire_refined = questionnaire_df.rdd.filter(v => v(1).toString.length == 1).map(v => (v(0).toString, v(1).toString))
    val questionnaire_count = questionnaire_df.count()
    val questionnaire_refined_count = questionnaire_refined.count()
    println("\n\n********************** questionnaire_count: " + questionnaire_count + " ********************")
    println("********************** questionnaire_refined_count: " + questionnaire_refined_count + " ********************\n\n")

    val questionnaire_refined_summary = questionnaire_refined.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    questionnaire_refined_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))

    val questionnaire_refined_label = questionnaire_refined.map(v => {
      var label: Int = 0
      if(v._2 == "A")
        label = 2
      else if(v._2 == "B")
        label = 2
      else if(v._2 == "C")
        label = 2
      else if(v._2 == "D")
        label = 2
      else if(v._2 == "E")
        label = 2
      else if(v._2 == "F")
        label = 2
      else if(v._2 == "H")
        label = 1
      else label = 3
      (v._1, label)
    }).filter(_._2 !=0).filter(_._2 != 3).reduceByKey((a, b) => a)

    val questionnaire_refined_label_summary = questionnaire_refined_label.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    questionnaire_refined_label_summary.sortByKey().collect().foreach(v => println(v._1 + " " + v._2.toString))
    return questionnaire_refined_label.map(v => (v._1, v._2-1))
  }

  def get_child_stage_label_from_questionnaire(hiveContext: HiveContext, questionnaire_table_name: String): RDD[(String, Int)] = {
    val questionnaire_select_sql: String = "SELECT imei, Q7 from " + questionnaire_table_name + " where Q7!=\"\""
    val questionnaire_df = hiveContext.sql(questionnaire_select_sql)
    val questionnaire_refined = questionnaire_df.rdd.filter(v => v(1).toString.length == 1).map(v => (v(0).toString, v(1).toString))
    val questionnaire_count = questionnaire_df.count()
    val questionnaire_refined_count = questionnaire_refined.count()
    println("\n\n********************** questionnaire_count: " + questionnaire_count + " ********************")
    println("********************** questionnaire_refined_count: " + questionnaire_refined_count + " ********************\n\n")

    val questionnaire_refined_summary = questionnaire_refined.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    questionnaire_refined_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))

//    val questionnaire_refined_label = questionnaire_refined.map(v => {
//      var label: Int = 0
//      if(v._2 == "A")
//        label = 1
//      else if(v._2 == "B")
//        label = 2
//      else if(v._2 == "C")
//        label = 3
//      else if(v._2 == "D")
//        label = 4
//      else if(v._2 == "E")
//        label = 5
//      else if(v._2 == "F")
//        label = 6
//      else if(v._2 == "H")
//        label = 7
//      else label = 8
//      (v._1, label)
//    })

    val questionnaire_refined_label = questionnaire_refined.map(v => {
      var label: Int = 0
      if(v._2 == "A")
        label = 1
      else if(v._2 == "B")
        label = 1
      else if(v._2 == "C")
        label = 1
      else if(v._2 == "D")
        label = 2
      else if(v._2 == "E")
        label = 2
      else if(v._2 == "F")
        label = 2
      else if(v._2 == "H")
        label = 7
      else label = 8
      (v._1, label)
    })

    val questionnaire_refined_label_summary = questionnaire_refined_label.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    questionnaire_refined_label_summary.sortByKey().collect().foreach(v => println(v._1 + " " + v._2.toString))
    return questionnaire_refined_label.filter(_._2 !=0).filter(_._2 != 7).filter(_._2 != 8).map(v => (v._1, v._2-1)).reduceByKey((a, b) => a)
  }

  def get_child_stage_label_from_xxx_data(hiveContext: HiveContext, yestoday_Date: String): RDD[(String, Int)] = {
    val xxx_child_stage_sql: String = "select imei, child_stage from algo.yf_xxx_labels where child_stage=\"婴幼儿\" or child_stage=\"孕育期\" or child_stage=\"青少年\""
    val xxx_child_stage_labeled_data = hiveContext.sql(xxx_child_stage_sql).rdd.map(v => {
      var label: Int = 0
      if(v(1).toString == "婴幼儿")
        label = 1
      else if(v(1).toString == "孕育期")
        label = 2
      else if(v(1).toString == "青少年")
        label = 3
      (v(0).toString, label)
    }).filter(_._2 != 0).reduceByKey((a, b) => a)

    val xxx_child_stage_labeled_data_summary = xxx_child_stage_labeled_data.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    xxx_child_stage_labeled_data_summary.sortByKey().collect().foreach(v => println(v._1 + " " + v._2.toString))

    return xxx_child_stage_labeled_data.map(v => (v._1, v._2-1))
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

  def get_data_set_for_build_model(labeled_dataset: RDD[(String, Int)],
                                   user_behavoir_features_table_name: String,
                                   feature_dim: Int,
                                   balance: Boolean,
                                   hiveContext: HiveContext,
                                   yestoday_Date: String): (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)]) = {
    // get  latest date of user behavior features
    // val select_latest_date_sql = "show PARTITIONS " + user_behavoir_features_table_name
    // val latest_date: String = hiveContext.sql(select_latest_date_sql).map(v => v(0).toString.split("=")(1).toInt).collect().sortWith((a, b) => a > b)(0).toString
    println("\n\n***************** get_data_set_for_build_model  ************* ")
    println("***************** The date of user behavior: " + yestoday_Date + " *************")
    // user behavior features
    val select_imei_feature_sql: String = "select * from " + user_behavoir_features_table_name + " where stat_date=" + yestoday_Date
    val imei_feature_rdd: RDD[(String, String)] = hiveContext.sql(select_imei_feature_sql).rdd.map(v => (v(0).toString, v(1).toString))
    println("********************* The number of user hehavior features data: " + imei_feature_rdd.count() + " ***********************")
    println("********************* The number of labeled_dataset: " + labeled_dataset.count() + " ***********************")
    // building the train and predict set
    val predict_set_rdd: RDD[(String, LabeledPoint)] = imei_feature_rdd.subtractByKey(labeled_dataset).map(v => {
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

    val train_set_rdd: RDD[(String, LabeledPoint)] = imei_feature_rdd.join(labeled_dataset).map(v => {
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

  def build_model(data_set: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)]), classes_num: Int): (LogisticRegressionModel, RDD[(String, (Double, Double))]) = {
    println("\n\n********************* Build Model *************************")
    val trainSet: RDD[(String, LabeledPoint)] = data_set._1 // balance dataset

    val rdd_temp: Array[RDD[(String, LabeledPoint)]] = trainSet.randomSplit(Array(0.8, 0.2), 1234L)
    val train_rdd: RDD[(String, LabeledPoint)] = rdd_temp(0).cache()
    val valid_rdd: RDD[(String, LabeledPoint)] = rdd_temp(1).cache()
    println("********************* train set number: " + train_rdd.count() + " *************************")
    for(i <- 0 until classes_num){
      println("********************* label_"+ i.toString + ": " + train_rdd.filter(_._2.label == i).count() + " *******************")
    }
    println()
    println("********************* valid set number: " + valid_rdd.count() + " *************************")
    for(i <- 0 until classes_num){
      println("********************* label_"+ i.toString + ": " + valid_rdd.filter(_._2.label == i).count() + " *******************")
    }
    println()
    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(classes_num).run(train_rdd.map(_._2))
    val valid_result: RDD[(String, (Double, Double))] = valid_rdd.map(v => (v._1, (model.predict(v._2.features), v._2.label)))
    return (model, valid_result)
  }

  def predict_marriage(model: LogisticRegressionModel, dataset: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)])): RDD[(String, String)] = {
    val prediction: RDD[(String, String)] = dataset._2.map(v => {
      if (model.predict(v._2.features) == 1d)
        (v._1, "married")
      else
        (v._1, "unmarried")
    })
    val pre: RDD[(String, String)] = dataset._1.map(v => (v._1, if (v._2.label == 1d) "married" else "unmarried")).union(prediction)
    val pre_summary = pre.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    pre_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    return pre
  }

  def predict_parent(model: LogisticRegressionModel, dataset: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)])): RDD[(String, String)] = {
    val prediction: RDD[(String, String)] = dataset._2.map(v => {
      if (model.predict(v._2.features) == 1d)
        (v._1, "parent")
      else
        (v._1, "non_parent")
    })
    val pre: RDD[(String, String)] = dataset._1.map(v => (v._1, if (v._2.label == 1d) "parent" else "non_parent")).union(prediction)
    val pre_summary = pre.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    pre_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    return pre
  }

  def predict_child_stage(model: LogisticRegressionModel, dataset: (RDD[(String, LabeledPoint)], RDD[(String, LabeledPoint)])): RDD[(String, String)] = {
    val prediction: RDD[(String, String)] = dataset._2.map(v => {
      var res: String = ""
      if (model.predict(v._2.features) == 0d)
        res = "baby"
      else if (model.predict(v._2.features) == 1d)
        res = "pregnant"
      else res = "teen"
      (v._1, res)
    }).filter(_._2 != "")

    val pre: RDD[(String, String)] = dataset._1.map(v => {
      var temp = ""
      if (v._2.label == 0d)
        temp = "baby"
      else if (v._2.label == 1d)
        temp = "pregnant"
      else temp = "teen"
      (v._1, temp)
    }).filter(_._2 != "").union(prediction)

    val pre_summary = pre.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size))
    println()
    pre_summary.collect().foreach(v => println(v._1 + " " + v._2.toString))
    return pre
  }

  def ROC(valid_result: RDD[(Double, Double)]): (Double, Double) = {
    val binary_class_metrics = new BinaryClassificationMetrics(valid_result)
    val roc = binary_class_metrics.roc()
    val au_roc = binary_class_metrics.areaUnderROC()
    val accuracy = valid_result.filter(v => v._1 == v._2).count() * 1.0 / valid_result.count()
    // println("\n\n ********************** AUROC: " + au_roc + " ********************* \n\n")
    return (au_roc, accuracy)
  }

  def confusion_matrix(valid_result: RDD[(Double, Double)]): MulticlassMetrics = {
    val multiclass_metrics = new MulticlassMetrics(valid_result)
    return multiclass_metrics
  }

  def parent_data_invalidation_with_questionnaire_and_xxx(hiveContext: HiveContext, questionnaire_table_name: String): Unit = {
    // parent data invalidation
    val parent_xxx = get_parent_label_from_xxx_data(hiveContext)
    val parent_questionnaire = get_parent_label_from_questionnaire(hiveContext, questionnaire_table_name)
    val parent_imei_matched = parent_xxx.join(parent_questionnaire)
    val parent_imei_and_label_matched = parent_imei_matched.filter(v => (v._2._1 == v._2._2))
    val parent_imei_and_label_conflicted = parent_imei_matched.filter(v => (v._2._1 != v._2._2))
    println("\n\n******************** parent_data_invalidation_with_questionnaire_and_xxx *******************")
    println("******************** parent_imei_matched: " + parent_imei_matched + " *******************")
    println("******************** parent_imei_and_label_matched: " + parent_imei_and_label_matched + " *******************")
    println("******************** parent_imei_and_label_conflicted: " + parent_imei_and_label_conflicted + " *******************\n\n")
  }

  def merge_predictions(marriage: RDD[(String, String)], parent: RDD[(String, String)], child_stage: RDD[(String, String)]): RDD[(String, String, String, String)] = {
    val marriage_refine = marriage.reduceByKey((a, b) => a)
    val parent_refine = parent.reduceByKey((a, b) => a)
    val child_stage_refine = child_stage.reduceByKey((a, b) => a)
    println("\n\n******************** merge the predictions **************************")
    println("******************** marriage_count: " + marriage.count() + " marriage_refine_count: " + marriage_refine.count())
    println("******************** parent_count: " + parent.count() + " parent_refine_count: " + parent_refine.count())
    println("******************** child_stage_count: " + child_stage.count() + " child_stage_refine_count: " + child_stage_refine.count())
    val merged_predictions = marriage_refine.join(parent_refine).join(child_stage_refine).map(v => (v._1, v._2._1._1, v._2._1._2, v._2._2))
    println("******************** merged_predictions_count: " + merged_predictions.count())
    println()

    // refine among the result of marriage, parent and child_stage
    val merged_pred_refined_1 = merged_predictions.map(v => {
      val imei = v._1
      val m = v._2
      var p = v._3
      var c = v._4
      if (v._2 == "unmarried" && v._3 == "parent"){
        p = "non_parent"
        c = "null"
      }
      if (v._3 == "non_parent")
        c = "null"
      (imei, m, p, c)
    })

    merged_pred_refined_1.map(v => (v._2, v._1)).groupByKey().map(v => (v._1, v._2.size)).collect().foreach(v => println(v._1 + " " + v._2.toString))
    println()
    merged_pred_refined_1.map(v => (v._3, v._1)).groupByKey().map(v => (v._1, v._2.size)).collect().foreach(v => println(v._1 + " " + v._2.toString))
    println()
    merged_pred_refined_1.map(v => (v._4, v._1)).groupByKey().map(v => (v._1, v._2.size)).collect().foreach(v => println(v._1 + " " + v._2.toString))
    println()

    val marriage_labels = Array("married", "unmarried")
    val parent_labels = Array("parent", "non_parent")
    val child_stage_labels = Array("baby", "teen", "pregnant", "null")
    for(i <- 0 to 1)
      for(j <- 0 to 1)
        for(k <- 0 to 3)
          println("********* " + marriage_labels(i) + " && " + parent_labels(j) + " && " + child_stage_labels(k) + "      " + merged_pred_refined_1.filter(_._2 == marriage_labels(i)).filter(_._3 == parent_labels(j)).filter(_._4 == child_stage_labels(k)).count())

    return merged_pred_refined_1
  }

  def save_to_db(hiveContext: HiveContext, pred: RDD[(String, String, String, String)], table_name: String, yestoday_Date: String): Unit = {
    import hiveContext.implicits._
    val pred_df: DataFrame = pred.map(v => Imei_MPC(v._1, v._2, v._3, v._4)).toDF
    println("\n\n ********************* (Strarting)Insert result to yf table *********************")
    pred_df.registerTempTable("prediction")
    val create_sql: String = "create table if not exists " + table_name + " (imei string, marriage string, parent string, child_stage string) partitioned by (stat_date string) stored as textfile"
    val insert_sql: String = "insert overwrite table " + table_name + " partition(stat_date=" + yestoday_Date + " ) select * from prediction"
    hiveContext.sql(create_sql)
    hiveContext.sql(insert_sql)
    println("********************* (Done)Insert result to yf table *********************\n\n ")
  }
}

