### [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)

[Data](https://www.kaggle.com/c/dog-breed-identification/data)

| Model               | Score          | 
| ------------------- |:--------------:|
| ResNet50            | 0.70079        |
| Inception_ResNet_v2 | 0.29297        |

Data => Pre-trained models => Linear model => Results

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 8, 8, 1536)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1536)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               393472    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 120)               30840     
=================================================================
Total params: 424,312
Trainable params: 424,312
Non-trainable params: 0
_________________________________________________________________

Train on 8221 samples, validate on 2001 samples
Epoch 1/1
8221/8221 [==============================] - 6s 675us/step - loss: 0.4873 - acc: 0.8640 - val_loss: 0.3043 - val_acc: 0.9110
```

Use pre-trained models to extract features (the last conv layer results). 90% for 120 classes.