const std = @import("std");
const mnist_data_utils = @import("mnist_data_utils.zig");

// Function to calculate Euclidean distance between all the pixels in an image
fn distance_between_images(training_image: mnist_data_utils.RawImageData, test_image: mnist_data_utils.RawImageData) u64 {
    var sum: u64 = 0;
    for (training_image, test_image) |training_pixel, test_image_pixel| {
        sum += @as(u64, @intCast(
            std.math.pow(i64, @as(i64, training_pixel) - test_image_pixel, 2),
        ));
    }

    return std.math.sqrt(sum);
}

const LabeledDistance = struct {
    label: mnist_data_utils.LabelType,
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

// Based on https://stackoverflow.com/a/8545681/796832
fn getMostFrequentLabel(labeled_distances: []LabeledDistance) mnist_data_utils.LabelType {
    // First sort by the label so we can just use a single loop to find the most frequent label.
    // This will get the list in order like 1, 1, 1, 2, 2, 3, 3, 3, 3, 3.
    std.mem.sort(
        LabeledDistance,
        labeled_distances,
        {},
        LabeledDistance.sort_label_ascending,
    );

    var most_popular_label_count: u32 = 0;
    var most_popular_label: mnist_data_utils.LabelType = 0;

    var previous_label: mnist_data_utils.LabelType = labeled_distances[0].label;
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

pub const PredictionResult = struct {
    prediction: mnist_data_utils.LabelType,
    debug: struct {
        neighbors: []const LabeledDistance,
    },
};

// (knn). Just straight up compares all of the pixels in each of the training images to
// the sample to see which one is the closest. The K number of closest images are
// grouped together and the most frequent label in the group is chosen as the
// prediction.
//
// With many training images, this is very slow. For example, if there are 60k training
// images, we compare every pixel of those 60k images to the test sample image.
pub fn kNearestNeighbors(
    training_images: []const mnist_data_utils.RawImageData,
    training_labels: []const mnist_data_utils.LabelType,
    test_image: mnist_data_utils.RawImageData,
    k: u8,
    allocator: std.mem.Allocator,
) !PredictionResult {
    const labeled_distances = try allocator.alloc(LabeledDistance, training_images.len);
    defer allocator.free(labeled_distances);

    // Compare our test image against all of the training images to see which ones are the closest
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
    const k_nearest_labeled_distances = try allocator.alloc(LabeledDistance, k);
    // We make a copy instead of just slicing because we return it from the function
    // and we don't want to return stack memory that will be freed after the function returns.
    std.mem.copy(
        LabeledDistance,
        k_nearest_labeled_distances,
        labeled_distances[0..k],
    );
    // std.log.debug("k_nearest_labeled_distances {any}", .{k_nearest_labeled_distances});

    // Get the most frequent label from the nearest neighbors
    const most_frequent_label = getMostFrequentLabel(k_nearest_labeled_distances);
    // std.log.debug("most_frequent_label {any}", .{most_frequent_label});

    return .{
        .prediction = most_frequent_label,
        .debug = .{
            .neighbors = k_nearest_labeled_distances,
        },
    };
}
