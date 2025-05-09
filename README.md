# SOAP_MLPFF

1. create a new folder, copy files of "code" folder, and files of one dataset folder (e.g. Delaney100_763_Converged_HF_STO3G/) to the new foler.

2. Modify hyperparameters in train_it.py and run it to train the model.

3. Use analysis.ipynb to analyze the dataset and model predictions.

# Docker
We already built the docker image for both cpu and gpu version of our code for you to run on one button!

GPU:https://hub.docker.com/layers/chanlingbg/soap_mlpff/gpu_cu118_final_git/images/sha256-d7a831affaea9b51f5d1166257415d4c88d0734e5e6f7fa421420057dc573d5a

CPU:https://hub.docker.com/layers/chanlingbg/soap_mlpff/cpu_final2.0_git/images/sha256-fece507101c852ab821d67d88cb277a16a173786d823a41921c4cdc1fe95b8f1

