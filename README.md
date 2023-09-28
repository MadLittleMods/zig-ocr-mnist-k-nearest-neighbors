# Basic OCR example

Simple OCR project to detect text in images from the MNIST dataset which is just a bunch
of 28x28 images of white number digits centered on a black background.

We're using K-nearest neighbor to classify the images which is the simplest way we can
compare our test sample against our training samples. Basically it takes our test sample
and compares it to all the training samples and finds the closest match (yes, it is
inneficient and slow with a lot of training samples).

Following this tutorial: https://www.youtube.com/watch?v=vzabeKdW9tE



### MNIST dataset

Downloaded from http://yann.lecun.com/exdb/mnist/

> All the integers in the files are stored in the MSB first (high endian)

> The training set contains 60000 examples, and the test set 10000 examples.
>
> [...]
>
> The first 5000 are cleaner and easier than the last 5000.

> TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
> [offset] [type]          [value]          [description]
> 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
> 0004     32 bit integer  60000            number of items
> 0008     unsigned byte   ??               label
> 0009     unsigned byte   ??               label
> ........
> xxxx     unsigned byte   ??               label
> The labels values are 0 to 9.
> 
> TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
> [offset] [type]          [value]          [description]
> 0000     32 bit integer  0x00000803(2051) magic number
> 0004     32 bit integer  60000            number of images
> 0008     32 bit integer  28               number of rows
> 0012     32 bit integer  28               number of columns
> 0016     unsigned byte   ??               pixel
> 0017     unsigned byte   ??               pixel
> ........
> xxxx     unsigned byte   ??               pixel
> Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
> 
> TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
> [offset] [type]          [value]          [description]
> 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
> 0004     32 bit integer  10000            number of items
> 0008     unsigned byte   ??               label
> 0009     unsigned byte   ??               label
> ........
> xxxx     unsigned byte   ??               label
> The labels values are 0 to 9.
> 
> TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
> [offset] [type]          [value]          [description]
> 0000     32 bit integer  0x00000803(2051) magic number
> 0004     32 bit integer  10000            number of images
> 0008     32 bit integer  28               number of rows
> 0012     32 bit integer  28               number of columns
> 0016     unsigned byte   ??               pixel
> 0017     unsigned byte   ??               pixel
> ........
> xxxx     unsigned byte   ??               pixel
> Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
