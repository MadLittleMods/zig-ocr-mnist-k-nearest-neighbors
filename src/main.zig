const std = @import("std");
const mnist_data_utils = @import("mnist_data_utils.zig");
const k_nearest_neighbors = @import("k_nearest_neighbors.zig");
const print_utils = @import("print_utils.zig");

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

// Adjust as necessary. To make the program run faster, you can reduce the number of
// images to train on and test on. To make the program more accurate, you can increase
// the number of images to train on (also try playing with the value of `k` in the model
// which influences K-nearest neighbor algorithm).
const NUMBER_OF_IMAGES_TO_TRAIN_ON = 10000; // (max 60k)
const NUMBER_OF_IMAGES_TO_TEST_ON = 100; // (max 10k)

pub const PredictiveModel = struct {
    // XXX: Make sure to tune this value to fit the data better (play with the number
    // and see how it affects accuracy)
    k: u8 = 5,

    training_images: []const mnist_data_utils.RawImageData = undefined,
    training_labels: []const mnist_data_utils.LabelType = undefined,

    // Since we're just using K nearest neighbors (KNN), there's no upfront training we
    // can do. We just need to store the training data so we can use it to compare
    // against a test image when we later try to make a prediction.
    //
    // We could do some preprocessing here (called feature selection), like removing
    // pixels that don't contribute to giving us accurate results (less data would also
    // be faster to iterate over). See
    // https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877
    // for ideas.
    pub fn train(
        self: *@This(),
        training_images: []const mnist_data_utils.RawImageData,
        training_labels: []const mnist_data_utils.LabelType,
    ) !void {
        self.training_images = training_images;
        self.training_labels = training_labels;
    }

    pub fn predict(
        self: *@This(),
        test_image: mnist_data_utils.RawImageData,
        allocator: std.mem.Allocator,
    ) !k_nearest_neighbors.PredictionResult {
        return try k_nearest_neighbors.kNearestNeighbors(
            self.training_images,
            self.training_labels,
            test_image,
            self.k,
            allocator,
        );
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // Read in the MNIST training labels
    const training_labels_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistLabelFileHeader,
        mnist_data_utils.LabelType,
        TRAIN_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(training_labels_data.items);
    std.log.debug("training labels header {}", .{training_labels_data.header});
    try std.testing.expectEqual(training_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(training_labels_data.header.number_of_labels, 60000);

    // Read in the MNIST training images
    const training_images_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistImageFileHeader,
        mnist_data_utils.RawImageData,
        TRAIN_DATA_FILE_PATH,
        "number_of_images",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(training_images_data.items);
    std.log.debug("training images header {}", .{training_images_data.header});
    try std.testing.expectEqual(training_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(training_images_data.header.number_of_images, 60000);
    try std.testing.expectEqual(training_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(training_images_data.header.number_of_columns, 28);

    // Read in the MNIST testing labels
    const testing_labels_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistLabelFileHeader,
        mnist_data_utils.LabelType,
        TEST_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TEST_ON,
        allocator,
    );
    defer allocator.free(testing_labels_data.items);
    std.log.debug("testing labels header {}", .{testing_labels_data.header});
    try std.testing.expectEqual(testing_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(testing_labels_data.header.number_of_labels, 10000);

    // Read in the MNIST testing images
    const testing_images_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistImageFileHeader,
        mnist_data_utils.RawImageData,
        TEST_DATA_FILE_PATH,
        "number_of_images",
        NUMBER_OF_IMAGES_TO_TEST_ON,
        allocator,
    );
    defer allocator.free(testing_images_data.items);
    std.log.debug("testing images header {}", .{testing_images_data.header});
    try std.testing.expectEqual(testing_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(testing_images_data.header.number_of_images, 10000);
    try std.testing.expectEqual(testing_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(testing_images_data.header.number_of_columns, 28);

    // Setup our model
    var predictive_model = PredictiveModel{
        // XXX: We can tune this value to fit the data better (play with the number
        // and see how it affects accuracy)
        .k = 5,
    };
    // Since we're just using KNN, this is basically just a no-op (see docstring)
    try predictive_model.train(training_images_data.items, training_labels_data.items);

    // For debugging: look at a single image and its nearest neighbors
    // {
    //     const index_under_test: u32 = 0;
    //     const labeled_image_under_test = mnist_data_utils.LabeledImage{
    //         .label = testing_labels_data.items[index_under_test],
    //         .image = mnist_data_utils.Image{
    //             .pixels = testing_images_data.items[index_under_test],
    //         },
    //     };

    //     const prediction_result = try predictive_model.predict(labeled_image_under_test.image.pixels, allocator);
    //     defer allocator.free(prediction_result.debug.neighbors);
    //     std.log.debug("prediction {}", .{prediction_result.prediction});
    //     std.log.debug("nearest neighbors {any}", .{prediction_result.debug.neighbors});
    //     try print_utils.printLabeledImage(labeled_image_under_test, allocator);
    // }

    // Go through all the test images and see how many we get right
    var incorrect_prediction_count: u32 = 0;
    for (testing_images_data.items, testing_labels_data.items, 0..) |test_image, test_label, test_image_index| {
        const labeled_image_under_test = mnist_data_utils.LabeledImage{
            .label = test_label,
            .image = mnist_data_utils.Image{
                .pixels = test_image,
            },
        };

        // Since this whole process can take a while to run, give some indication of
        // progress while it runs.
        if (test_image_index % 100 == 0) {
            std.log.debug("Progress: working on test image {d}", .{test_image_index});
        }

        const prediction_result = try predictive_model.predict(labeled_image_under_test.image.pixels, allocator);
        defer allocator.free(prediction_result.debug.neighbors);
        // Only print when we get something wrong
        if (prediction_result.prediction != labeled_image_under_test.label) {
            incorrect_prediction_count += 1;
            std.log.debug("Test image {d}: incorrect prediction {}", .{ test_image_index, prediction_result.prediction });
            try print_utils.printLabeledImage(labeled_image_under_test, allocator);
        }
    }

    std.log.debug("incorrect_prediction_count {d} out of {d} test images", .{
        incorrect_prediction_count,
        testing_images_data.items.len,
    });
    const inaccuracy: f32 = @as(f32, @floatFromInt(incorrect_prediction_count)) / @as(f32, @floatFromInt(testing_images_data.items.len));
    std.log.debug("accuracy {d}", .{1 - inaccuracy});
}
