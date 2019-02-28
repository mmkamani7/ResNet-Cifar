# ResNet-Cifar
ResNet for CIFAR with Estimator API and tf.keras.Model class

You can download each dataset and convert it to tfrecords using:
```cli
python generate_cifar_tfrecords.py --data-dir=./cifar10 --dataset cifar10
```

Then, you can run the ResNet model using:

```cli
python main.py --data-dir=./cifar10 \
               --job-dir=./log/cifar10 \
               --dataset cifa10 \
               --num-layers=20 \
               --version v1
```
