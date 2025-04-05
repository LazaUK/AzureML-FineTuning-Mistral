# Fine-Tuning Mistral-7B in Azure Machine Learning Studio

This repository documents the process of fine-tuning the Mistral-7B model using Azure Machine Learning (Azure ML) Studio. Azure ML Studio provides a streamlined environment for fine-tuning curated models from its model catalog. This guide walks you through the necessary steps, from environment setup to deploying the fine-tuned model.

> [!NOTE]
> The Python code and training datasets in this repo are adapted from Microsoft's [Azure Machine Learning examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/finetuning) repo.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Configuring Environment](#step-1-configuring-environment)
- [Step 2: Defining Source Model](#step-2-defining-source-model)
- [Step 3: Preparing Training and Validation Dataset](#step-3-preparing-training-and-validation-dataset)
- [Step 4: Fine-tuning Model](#step-4-fine-tuning-model)
- [Step 5: Deploying Fine-tuned Model to Online Endpoint](#step-5-deploying-fine-tuned-model-to-online-endpoint)

## Prerequisites

Before you begin, ensure you have the following prerequisites in place:
1. **Azure Subscription:** If you don't have one, you can create a free account.
2. **Azure Machine Learning Workspace:** Ensure you have the necessary permissions to create resources within the Azure ML workspace.
3. **Compute Cluster with GPU SKUs:** in [Azure ML Studio](https://ml.azure.com), create a compute cluster with GPU SKUs suitable for fine-tuning the Mistral-7B model. Please, refer to the Azure ML documentation for recommended GPU types and sizes.
4. **Managed Identity:** During setup of compute cluster, assign a managed identity (MI) to it as shown below. This MI should be also assigned *Storage Blob Data Contributor* and *Storage File Data Privileged Contributor* to Azure Storage account, that will host training and validation datasets.
![Step0_Identity_Compute](images/Step0_Compute_Identity.png)
5. **Python Packages:** Install the required Python packages:
```powershell
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```

## Step 1: Configuring Environment

1. Set up your environment variables to make provided Jupyter notebook work.

| Variable                  | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `subscription_id`         | Azure subscription ID.                          |
| `resource_group`          | Azure ML resource group name.                     |
| `workspace_name`          | Azure ML workspace name.                         |
| `model_registry`          | Azure ML model registry name.                    |
| `model_name`              | Name of the model (e.g., `mistral-7b`).           |
| `training_dataset_name`   | Name of training dataset in Azure ML datastore. |
| `training_dataset_file`   | Filename of training dataset. |
| `validation_dataset_name` | Name of validation dataset in Azure ML datastore.|
| `validation_dataset_file` | Filename of validation dataset. |
| `dataset_version`         | Version of the dataset.                            |
| `job_name`                | Name of the fine-tuning job.                        |
| `job_compute`             | Name of the compute cluster for the job.         |
| `endpoint_name`           | Name of the online endpoint.                           |
| `endpoint_SKU`            | SKU for the online endpoint deployment.          |
| `guid`                    | Unique identifier. |

2. Authenticate to your Azure environment:
``` Python
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()
```
3. Initialise MLClient for both Azure ML workspace and Azure ML model registry:
``` Python
workspace_ml_client = MLClient(
    credential = credential,
    subscription_id = subscription_id,
    resource_group_name = resource_group,
    workspace_name = workspace_name,
)

registry_ml_client = MLClient(
    credential = credential,
    registry_name = model_registry
)
```
4. 
