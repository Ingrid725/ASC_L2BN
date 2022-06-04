# Acoustic Scene Classification with Multiple Devices

## Introduction
In order to further explore the effectiveness of our method, L2BN, we conduct it on the acoustice scene classification task. We experiment on the TUT Urban Acoustic Scenes 2020 Mobile Development dataset, which consists of 10-seconds audio segments from 10 acoustic scenes and contains in total of 64 hours of audio.  The task we choose is a subtask of the acoustic scene classification in the challenge on detection and classification of acoustic scenes and events (DCASE). The goal is to classify the audio recorded with multiple (real and simulated) devices into ten different fine-grained classes, including airport, public square and urban park, etc.

We reference [code](https://github.com/MihawkHu/DCASE2020_task1), a work on IEEE ICASSP 2021. It tests three CNN models on this task, and we use it to compare model performances with and without L2BN.

## How to use this code
Firstly, download the [DCASE 2020 task 1a development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-a). And then unzip the dataset under the data folder.
Then, run the file extr_feat_2020_nodelta_scaled.py to generate logmel features of sound files.
```
python extr_feat_2020_nodelta_scaled.py
```
<<<<<<< HEAD
<<<<<<< HEAD
Then, go to the specific model's folder, and train the model with following command.
=======
Then, go to the specific model's folder, and run the model with following command.
>>>>>>> f2085b458465827941065feefedb2251ccc1e4c6
=======
Then, go to the specific model's folder, and run the model with following command.
>>>>>>> f2085b458465827941065feefedb2251ccc1e4c6
```
cd resnet
CUDA_VISIBLE_DEVICES=0,1 python train_resnet.py
CUDA_VISIBLE_DEVICES=0,1 python train_resnet.py --l2bn
```
<<<<<<< HEAD
<<<<<<< HEAD
And we also supply some pretrained models, you can evaluate it with following command.
```
python eval_model_10class.py  
```
=======
>>>>>>> f2085b458465827941065feefedb2251ccc1e4c6
=======
>>>>>>> f2085b458465827941065feefedb2251ccc1e4c6

## Experiment results
Tested on [DCASE 2020 task 1a development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-a). The train-test split way follows the official recomendation.  

| Model      |   BN/L2BN  |   Accuracy(%) | 
| :---       |   :----:   |      :----:   | 
|  Resnet  | BN     | 72.51%    |
|  Resnet  | L2BN   | 72.64%    |  
|  FCNN    | BN     | 69.17%    | 
|  FCNN    | L2BN   | 70.55%    | 
|  fsFCNN  | BN     | 71.19%    | 
|  fsFCNN  | L2BN   | 72.44%    |
