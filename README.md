# Fine-Tuning Mistral-7B in Azure Machine Learning Studio

This repository documents the process of fine-tuning the Mistral-7B model using Azure Machine Learning (Azure ML) Studio. Azure ML Studio provides a streamlined environment for fine-tuning curated models from its model catalog. This guide walks you through the necessary steps, from environment setup to deploying the fine-tuned model.

> [!NOTE]
> The Python code and training datasets in this repository are adapted from Microsoft's [Azure Machine Learning examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/finetuning) repository.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Configuring Environment](#step-1-configuring-environment)
- [Step 2: Defining Source Model](#step-2-defining-source-model)
- [Step 3: Preparing Training and Validation Dataset](#step-3-preparing-training-and-validation-dataset)
- [Step 4: Fine-tuning Model](#step-4-fine-tuning-model)
- [Step 5: Deploying Fine-tuned Model to Online Endpoint](#step-5-deploying-fine-tuned-model-to-online-endpoint)

## Prerequisites

Before you begin, ensure you have the following prerequisites in place:

1.  **Azure Subscription:** You need an active Azure subscription. If you don't have one, you can create a free account.
2.  **Azure Machine Learning Workspace:** Create an Azure ML workspace in Azure / ensure you have the necessary permissions to create resources within the workspace.
3.  **Compute Cluster with GPU SKUs:** in Azure ML Studio, create a compute cluster with GPU SKUs suitable for fine-tuning the Mistral-7B model (e.g., NVIDIA A100, V100). Please, refer to the Azure ML documentation for recommended GPU types and sizes.
4.  **Python Packages:** Install the required Python packages:
```powershell
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```

## Step 1: Configuring Environment

Set up your environment variables to connect to your Azure ML workspace.

| Variable              | Description                                      |
| --------------------- | ------------------------------------------------ |
| `subscription_id`     | Azure subscription ID.                          |
| `aml_staging_name`    | Azure ML staging workspace name.              |
| `aml_staging_rg`      | Azure ML staging resource group name.          |
| `aml_production_name` | Azure ML production workspace name (optional). |
| `aml_production_rg`   | Azure ML production resource group name (optional).|
| `aml_registry_name`   | Azure ML registry name.                           |
| `aml_registry_location`| Azure ML registry location.                        |
| `model_name`          | Name of the model (e.g., `mistral-7b`).           |
| `model_path`          | Path to the model within the registry (if applicable). |
| `model_version`       | Version of the model.                               |

**Python Code Snippet (Example):**
