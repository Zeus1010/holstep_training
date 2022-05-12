# CIS 700 Project : Training on HolStep Dataset
### After downloading the file make sure that you have installed all necessary packages.
### Packages can be downloaded using the following command:
```
pip3 install -r requirements.txt
```
There are two python files, both implementing two different models :
1. Basic 2 Layered Convolutional Neural Network
2. Basic 2 Layered Convolutional Neural Network along with LSTM

The values following 5 params has not been set according to the basic norm, since it was not possible to train this much amount on data on the current system. Hence if you want you can change these values thereby increasing the performance to a greater extent.
1. training_batch_size = 64
2. training_max_len = 512
3. epochs = 10
4. steps_per_epoch = 100
5. validation_steps = 10

***If you have downloaded this code, pleasae change the name of parent directory to holstep_training, since after downloading it will be holstep_training-master.***

### For implementing the first model(cnn), type in the following command :
```
python3 model_1.py 
```
### For implementing the second model(cnn_lstm), type in the following command :
```
python3 model_2.py 
```
