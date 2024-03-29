# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC 
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC 
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC 
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC 
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC 
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC 
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC 
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

cluster_json = {
    "num_workers": 4,
    "cluster_name": "sparkNLP_FSI_v4",
    "spark_version": "9.1.x-cpu-ml-scala2.12", # This needs to match JSL version in partner connect
    "spark_conf": {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "2000M",
        "spark.databricks.delta.formatCheck.enabled": "false"
    },
    "node_type_id": {"AWS": "i3.2xlarge", "MSA": "Standard_DS4_v2", "GCP": "n1-highmem-4"}, # different from standard API; this is multi-cloud friendly
    "autotermination_minutes": 120
}

# COMMAND ----------

nsc = NotebookSolutionCompanion()
cluster_id = nsc.create_or_update_cluster_by_name(nsc.customize_cluster_json(cluster_json))

# COMMAND ----------

task_json = {'tasks': [{
    'task_key': 'setup_cluster',
    'depends_on': [],
    'existing_cluster_id': cluster_id,
    "notebook_task": {
        "notebook_path": "/Shared/John Snow Labs/Install JohnSnowLabs NLP",
        "source": "WORKSPACE"
        },
    'timeout_seconds': 86400}]
            }
nsc.submit_run(task_json)

# COMMAND ----------

job_json = {
        "timeout_seconds": 7200,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "FSI"
        },
        "tasks": [
            {
                "existing_cluster_id": cluster_id,
                "notebook_task": {
                    "notebook_path": "01_Introduction_And_Setup"
                },
                "task_key": "jsl_fin_01",
                "description": ""
            },
            {
                "existing_cluster_id": cluster_id,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "02_10K_Analysis"
                },
                "task_key": "jsl_fin_02",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "jsl_fin_01"
                    }
                ]
            },
            {
                "existing_cluster_id": cluster_id,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "03_Named_Entity_Recognition"
                },
                "task_key": "jsl_fin_03",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "jsl_fin_02"
                    }
                ]
            },
            {
                "existing_cluster_id": cluster_id,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "04_Normalization_Data_Augmentation"
                },
                "task_key": "jsl_fin_04",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "jsl_fin_03"
                    }
                ]
            },
            {
                "existing_cluster_id": cluster_id,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "05_Relation_Extraction"
                },
                "task_key": "jsl_fin_05",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "jsl_fin_04"
                    }
                ]
            },
            {
                "existing_cluster_id": cluster_id,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "06_Understanding_Entities_in_Context"
                },
                "task_key": "jsl_fin_06",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "jsl_fin_05"
                    }
                ]
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc.deploy_compute(job_json, run_job=run_job, wait=3600)

# COMMAND ----------



# COMMAND ----------


