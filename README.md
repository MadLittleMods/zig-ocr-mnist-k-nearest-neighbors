# Basic OCR example using K-nearest neighbors against the MNIST dataset

A from scratch, simple OCR project to recognize/detect text in images from the MNIST
dataset which is just a bunch of 28x28 images of white number digits centered on a black
background.

We're using K-nearest neighbors to classify the images which is the simplest way we can
compare our test sample against our training samples. Basically it takes our test sample
image and compares it to all the training samples and finds the closest match (yes, it
is inneficient and slow with a lot of training samples).

Basically a Zig rewrite following this tutorial by Vlad Harbuz from @clumsycomputer:
https://www.youtube.com/watch?v=vzabeKdW9tE.

Just getting my feet wet in OCR.

## Setup

Download and extract the MNIST dataset from http://yann.lecun.com/exdb/mnist/ to a
directory called `data/` in the root of this project. Here is a copy-paste command
you can run:

```sh
# Make a data/ directory
mkdir data/ &&
cd data/ &&
# Download the MNIST dataset
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz &&
# Unzip the files
gunzip *.gz
```


## Building and running

Tested with Zig 0.11.0

```sh
$ zig build run
zig build run
debug: training labels header mnist_data_utils.MnistLabelFileHeader{ .magic_number = 2049, .number_of_labels = 60000 }
debug: training images header mnist_data_utils.MnistImageFileHeader{ .magic_number = 2051, .number_of_images = 60000, .number_of_rows = 28, .number_of_columns = 28 }
debug: testing labels header mnist_data_utils.MnistLabelFileHeader{ .magic_number = 2049, .number_of_labels = 10000 }
debug: testing images header mnist_data_utils.MnistImageFileHeader{ .magic_number = 2051, .number_of_images = 10000, .number_of_rows = 28, .number_of_columns = 28 }
debug: prediction 7
debug: nearest neighbors { k_nearest_neighbors.LabeledDistance{ .label = 7, .distance = 1034 }, k_nearest_neighbors.LabeledDistance{ .label = 7, .distance = 1047 }, k_nearest_neighbors.LabeledDistance{ .label = 7, .distance = 1095 }, k_nearest_neighbors.LabeledDistance{ .label = 7, .distance = 1097 }, k_nearest_neighbors.LabeledDistance{ .label = 7, .distance = 1121 } }
┌──────────┐
│ Label: 7 │
┌────────────────────────────────────────────────────────┐
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│            ▒▒▓▓▓▓▓▓░░░░                                │
│            ████████████████████████████▓▓░░            │
│            ▒▒▒▒▒▒▒▒▓▓████████████████████▓▓            │
│                      ░░▒▒░░▒▒▒▒▒▒░░░░████▒▒            │
│                                    ▒▒████░░            │
│                                  ░░████▒▒              │
│                                  ▓▓████░░              │
│                                ░░████░░                │
│                                ▓▓██▓▓░░                │
│                              ░░████░░                  │
│                              ▒▒██▓▓                    │
│                            ▒▒████░░                    │
│                          ░░████▓▓                      │
│                        ░░██████░░                      │
│                        ░░████▒▒                        │
│                      ░░████▒▒░░                        │
│                      ▓▓████░░                          │
│                    ░░██████░░                          │
│                    ▒▒██████░░                          │
│                    ▒▒████░░                            │
│                                                        │
└────────────────────────────────────────────────────────┘
...
```


## Results/Accuracy

With `k=5`, running with all 60k training images against the 10k test images, we get
an accuracy of 96.72% (9672/10000).

With `k=10`, running with all 60k training images against the 10k test images, we get
an accuracy of 95.09% (9509/10000).

## Dev notes

See the [*developer notes*](./dev-notes.md) for more information.

