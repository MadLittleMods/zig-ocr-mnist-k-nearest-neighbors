# Basic OCR example using K-nearest neighbor against the MNIST dataset

A from scratch, simple OCR project to detect text in images from the MNIST dataset which
is just a bunch of 28x28 images of white number digits centered on a black background.

We're using K-nearest neighbor to classify the images which is the simplest way we can
compare our test sample against our training samples. Basically it takes our test sample
image and compares it to all the training samples and finds the closest match (yes, it
is inneficient and slow with a lot of training samples).

Basically a Zig rewrite following this tutorial by Vlad Harbuz from @clumsycomputer:
https://www.youtube.com/watch?v=vzabeKdW9tE.

Just getting my feet wet in OCR.

## Setup

Download and extract the MNIST dataset from http://yann.lecun.com/exdb/mnist/ to a
directory called `data/` in the root of this project.

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

```
zig build run
```



### Dev notes

See the [*developer notes*](./dev-notes.md) for more information.

