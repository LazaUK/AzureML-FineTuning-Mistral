# FineTuning Mistral-7B in Azure Machine Learning studio

Azure ML studio enables fine-tuning of curated models from Azure Machine Learning's model catalog. This repo walks through the process of Mistral 7B model.

> [!NOTE]
> Python code and training datasets were adopted from Microsoft's [Azure Machine Learning examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/finetuning) repo.

## Table of contents:
- [Prerequistes]()
- [Step 1: Configuring Environment]()
- [Step 2: Defining Source Model]()
- [Step 3: Preparing Training and Validation Dataset]()
- [Step 4: Fine-tuning Model]()
- [Step 5: Deploying Finetuned Model to Online Endpoint]()

## Prerequistes
1. Install Azure ML SDK and Azure Identity Python packages:
``` PowerShell
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```
2. In [Azure ML studio](https://ml.azure.com) creation compute cluster with GPU SKUs, supported by the target model's finetuning process:

## Step 1: Configuring Environment

| Variable | Description |
| --- | --- |
| ```subscription_id``` | Azure subscription ID |
| ```aml_staging_name``` | AML staging workspace name |
| ```aml_staging_rg``` | AML staging resource group name |
| ```aml_production_name``` | AML production workspace name |
| ```aml_production_rg``` | AML production resource group name |
| ```aml_registry_name``` | AML registry name |
| ```aml_registry_location``` | AML registry location |
| ```model_name``` | model name |
| ```model_path``` | model path |
| ```model_version``` | model version |

