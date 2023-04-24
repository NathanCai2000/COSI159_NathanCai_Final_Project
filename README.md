# COSI 159A - MNIST Classification:

## Create Conda Environment:

You can follow the tutorial listed by Conda: [Managing Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
1. Create Environment

        conda create --name pytorchlrn

2. Activate the Environment

        conda activate pytorchlrn

3. Install torchvision and matplotlib

        pip install torchvision
        pip install matplotlib

## How to run Program:
1. Navigate to the Task_1 directory in Terminal
2. Run the main.py file

        python main.py
3. The terminal will list out the current model's averagelost rate and the accuracy rate.
4. The model and its optimizer will be stored in the '/results' folder in the Task_1 directory.

## Program Optional Arguements:
The listed arguments are all optional, allowing more control in the model construction.

| Name          | Description                                    | Inputs and Type      | Defaults |
| ------------- | ---------------------------------------------- | -------------------- | -------- |
| Epochs        | Change the number of training cycles           | '--epochs' = int     | 10       |
| Learning Rate | Changes the training learning rate             | '--lr' = flaot       | 0.1      |
| Batch Size    | Adjust the initial batch sizes from MNIST      | '--bs' = int         | 64       |