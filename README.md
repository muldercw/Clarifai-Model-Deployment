# Clarifai-Model-Deployment

https://www.loom.com/share/dd44ffa03c844a28a58d3e5d4806033e?sid=0c60f781-2347-4989-8204-7524e11b0cde


## Description
This project will export, format, deploy and test all models from a given App from the Clarifai platform. 
Utilizes a Docker container running Triton Server: 23.03-py3 

## Installation
Built on Windows 11 machine with following Cuda versions:

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:30:10_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

## Usage
Ensure you update the sample videos folder with mp4 videos you would like to test the model on. 
Ensure you have the correct user_id, pat and app_id for the Clarifai platform.

[`  python autoDeployModel.py "user_id" "pat" "app_id"  `]
