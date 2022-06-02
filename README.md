# Acoustic Scene Classification with Multiple Devices

## Introduction
In order to further explore the effectiveness of our method, L2BN, we conduct it on the acoustice scene classification task. We experiment on the TUT Urban Acoustic Scenes 2020 Mobile Development dataset, which consists of 10-seconds audio segments from 10 acoustic scenes and contains in total of 64 hours of audio.  The task we choose is a subtask of the acoustic scene classification in the challenge on detection and classification of acoustic scenes and events (DCASE). The goal is to classify the audio recorded with multiple (real and simulated) devices into ten different fine-grained classes, including airport, public square and urban park, etc.
We reference [code](https://github.com/MihawkHu/DCASE2020_task1), a work on IEEE ICASSP 2021. It tests three CNN models on this task, and we use it to compare model performances with and without L2BN.

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
