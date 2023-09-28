# Basic OCR example

Simple OCR project to detect text in images from the MNIST dataset which is just a bunch
of 28x28 images of white number digits centered on a black background.

We're using K-nearest neighbor to classify the images which is the simplest way we can
compare our test sample against our training samples. Basically it takes our test sample
and compares it to all the training samples and finds the closest match (yes, it is
inneficient and slow with a lot of training samples).

Following this tutorial: https://www.youtube.com/watch?v=vzabeKdW9tE


### Building and running

Tested with Zig 0.11.0

```
zig build run
```



### Dev notes

See the [*developer notes*](./dev-notes.md) for more information.

