const std = @import("std");
const bigEndianStructDeserializer = @import("big_endian_struct_deserializer.zig").bigEndianStructDeserializer;

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

const NUMBER_OF_IMAGES_TO_TRAIN_ON = 10000;

const MnistLabelFileHeader = extern struct {
    magic_number: u32,
    number_of_labels: u32,
};

const MnistImageFileHeader = extern struct {
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,
};

const RawImageData = [28 * 28]u8;

const Image = extern struct {
    width: u8 = 28,
    height: u8 = 28,
    pixels: RawImageData,
};

const LabelType = u8;
const LabeledImage = extern struct {
    label: LabelType,
    image: Image,
};

fn decorateStringWithAnsiColor(input_string: []const u8, hex_color: u24, allocator: std.mem.Allocator) ![]const u8 {
    const string = try std.fmt.allocPrint(
        allocator,
        "\u{001b}[38;2;{d};{d};{d}m{s}\u{001b}[0m",
        .{
            // red
            hex_color >> 16,
            // green
            (hex_color >> 8) & 0xFF,
            // blue
            hex_color & 0xFF,
            input_string,
        },
    );
    return string;
}

fn printImage(image: Image, allocator: std.mem.Allocator) !void {
    std.debug.print("┌", .{});
    for (0..(image.width - 1)) |column_index| {
        _ = column_index;
        std.debug.print("──", .{});
    }
    std.debug.print("┐\n", .{});

    for (0..(image.height - 1)) |row_index| {
        std.debug.print("│", .{});

        const row_start_index = row_index * image.width;
        for (0..(image.width - 1)) |column_index| {
            const index = row_start_index + column_index;
            const pixel_value = image.pixels[index];
            const colored_pixel_string = try decorateStringWithAnsiColor(
                "\u{2588}\u{2588}",
                // Create a white color with the pixel value as the brightness
                (@as(u24, pixel_value) << 16) | (@as(u24, pixel_value) << 8) | (@as(u24, pixel_value) << 0),
                allocator,
            );
            defer allocator.free(colored_pixel_string);
            std.debug.print("{s}", .{
                colored_pixel_string,
            });
        }
        std.debug.print("│\n", .{});
    }

    std.debug.print("└", .{});
    for (0..(image.width - 1)) |column_index| {
        _ = column_index;
        std.debug.print("──", .{});
    }
    std.debug.print("┘\n", .{});
}

fn printLabeledImage(labeled_image: LabeledImage, allocator: std.mem.Allocator) !void {
    std.debug.print("┌──────────┐\n", .{});
    std.debug.print("│ Label: {d} │\n", .{labeled_image.label});
    try printImage(labeled_image.image, allocator);
}

fn MnistData(comptime HeaderType: type, comptime ItemType: type) type {
    return struct {
        header: HeaderType,
        items: []const ItemType,
    };
}

fn readMnistFile(
    comptime HeaderType: type,
    comptime ItemType: type,
    file_path: []const u8,
    comptime numberOfItemsFieldName: []const u8,
    number_of_items_to_read: u32,
    allocator: std.mem.Allocator,
) !MnistData(HeaderType, ItemType) {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // Use a buffered reader for better performance (less syscalls)
    // (https://zig.news/kristoff/how-to-add-buffering-to-a-writer-reader-in-zig-7jd)
    var buffered_reader = std.io.bufferedReader(file.reader());
    var file_reader = buffered_reader.reader();

    const header = try file_reader.readStructBig(HeaderType);

    // Make sure we don't try to allocate more images than there are in the file
    std.debug.assert(@field(header, numberOfItemsFieldName) >= number_of_items_to_read);

    const image_data_array = try allocator.alloc(ItemType, number_of_items_to_read);
    const deserializer = bigEndianStructDeserializer(file_reader);
    for (0..(number_of_items_to_read - 1)) |image_index| {
        const image = try deserializer.read(ItemType);
        image_data_array[image_index] = image;
    }

    return .{
        .header = header,
        .items = image_data_array[0..],
    };
}

// Function to calculate Euclidean distance between all the pixels in an image
fn distance_between_images(training_image: RawImageData, test_image: RawImageData) u64 {
    var sum: u64 = 0;
    for (training_image, test_image) |training_pixel, test_image_pixel| {
        sum += @as(u64, @intCast(
            std.math.pow(i64, @as(i64, training_pixel) - test_image_pixel, 2),
        ));
    }

    return std.math.sqrt(sum);
}

const LabeledDistance = struct {
    label: LabelType,
    distance: u64,

    fn sort_label_ascending(context: void, lhs: @This(), rhs: @This()) bool {
        _ = context;
        return lhs.label < rhs.label;
    }

    fn sort_distance_ascending(context: void, lhs: @This(), rhs: @This()) bool {
        _ = context;
        return lhs.distance < rhs.distance;
    }
};

fn getMostFrequentLabel(labeled_distances: []LabeledDistance) LabelType {
    // First sort by the label so we can just use a single loop to find the most frequent label
    std.mem.sort(
        LabeledDistance,
        labeled_distances,
        {},
        LabeledDistance.sort_label_ascending,
    );

    var most_popular_label_count: u32 = 0;
    var most_popular_label: LabelType = 0;

    var previous_label: LabelType = labeled_distances[0].label;
    var current_label_count: u32 = 0;
    for (labeled_distances) |labeled_distance| {
        if (labeled_distance.label == previous_label) {
            current_label_count += 1;
        }

        if (current_label_count > most_popular_label_count) {
            most_popular_label_count = current_label_count;
            most_popular_label = labeled_distance.label;
        }

        previous_label = labeled_distance.label;
    }

    return most_popular_label;
}

const PredictionResult = struct {
    prediction: LabelType,
    debug: struct {
        neighbors: []const LabeledDistance,
    },
};

// (knn)
fn kNearestNeighbors(
    training_images: []const RawImageData,
    training_labels: []const LabelType,
    test_image: RawImageData,
    k: u8,
    allocator: std.mem.Allocator,
) !PredictionResult {
    const labeled_distances = try allocator.alloc(LabeledDistance, training_images.len);
    defer allocator.free(labeled_distances);

    for (training_images, training_labels, 0..) |training_image, training_label, training_index| {
        const training_distance = distance_between_images(training_image, test_image);

        labeled_distances[training_index] = LabeledDistance{
            .label = training_label,
            .distance = training_distance,
        };
    }

    // Sort the distances from closest to furthest (smallest to largest)
    std.mem.sort(
        LabeledDistance,
        labeled_distances,
        {},
        LabeledDistance.sort_distance_ascending,
    );

    // Find the k nearest neighbors
    const k_nearest_labeled_distances = labeled_distances[0..k];
    // std.log.debug("k_nearest_labeled_distances {any}", .{k_nearest_labeled_distances});

    // Get the most frequent label from the nearest neighbors
    const most_frequent_label = getMostFrequentLabel(k_nearest_labeled_distances);
    // std.log.debug("most_frequent_label {any}", .{most_frequent_label});

    return .{
        .prediction = most_frequent_label,
        .debug = .{
            // TODO: Accessing this will give a segmentation fault because `labeled_distances` is freed
            // after this function returns. Need to figure out how to make this work.
            .neighbors = k_nearest_labeled_distances,
        },
    };
}

pub const PredictiveModel = struct {
    training_images: []const RawImageData = undefined,
    training_labels: []const LabelType = undefined,

    pub fn train(
        self: *@This(),
        training_images: []const RawImageData,
        training_labels: []const LabelType,
    ) !void {
        self.training_images = training_images;
        self.training_labels = training_labels;
    }

    pub fn predict(
        self: *@This(),
        test_image: RawImageData,
        allocator: std.mem.Allocator,
    ) !PredictionResult {
        return try kNearestNeighbors(
            self.training_images,
            self.training_labels,
            test_image,
            5,
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

    const training_labels_data = try readMnistFile(
        MnistLabelFileHeader,
        u8,
        TRAIN_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(training_labels_data.items);
    std.log.debug("training labels header {}", .{training_labels_data.header});
    try std.testing.expectEqual(training_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(training_labels_data.header.number_of_labels, 60000);

    const training_images_data = try readMnistFile(
        MnistImageFileHeader,
        RawImageData,
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

    const testing_labels_data = try readMnistFile(
        MnistLabelFileHeader,
        u8,
        TEST_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(testing_labels_data.items);
    std.log.debug("testing labels header {}", .{testing_labels_data.header});
    try std.testing.expectEqual(testing_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(testing_labels_data.header.number_of_labels, 10000);

    const testing_images_data = try readMnistFile(
        MnistImageFileHeader,
        RawImageData,
        TEST_DATA_FILE_PATH,
        "number_of_images",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(testing_images_data.items);
    std.log.debug("testing images header {}", .{testing_images_data.header});
    try std.testing.expectEqual(testing_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(testing_images_data.header.number_of_images, 10000);
    try std.testing.expectEqual(testing_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(testing_images_data.header.number_of_columns, 28);

    var predictive_model = PredictiveModel{};
    try predictive_model.train(training_images_data.items, training_labels_data.items);

    // {
    //     const index_under_test: u32 = 5;
    //     const labeled_image_under_test = LabeledImage{
    //         .label = testing_labels_data.items[index_under_test],
    //         .image = Image{
    //             .pixels = testing_images_data.items[index_under_test],
    //         },
    //     };

    //     const prediction_result = try predictive_model.predict(labeled_image_under_test.image.pixels, allocator);
    //     std.log.debug("prediction {}", .{prediction_result.prediction});
    //     std.log.debug("nearest neighbors {any}", .{prediction_result.debug.neighbors});
    //     try printLabeledImage(labeled_image_under_test, allocator);
    // }

    for (testing_images_data.items, testing_labels_data.items, 0..) |test_image, test_label, test_image_index| {
        const labeled_image_under_test = LabeledImage{
            .label = test_label,
            .image = Image{
                .pixels = test_image,
            },
        };

        const prediction_result = try predictive_model.predict(labeled_image_under_test.image.pixels, allocator);

        if (prediction_result.prediction != labeled_image_under_test.label) {
            std.log.debug("{d}: prediction {}", .{ test_image_index, prediction_result.prediction });
            try printLabeledImage(labeled_image_under_test, allocator);
        }
    }
}
