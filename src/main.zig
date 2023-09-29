const std = @import("std");
const bigEndianStructDeserializer = @import("big_endian_struct_deserializer.zig").bigEndianStructDeserializer;

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

const NUMBER_OF_IMAGES_TO_TRAIN_ON = 100;

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

const LabeledImage = extern struct {
    label: u8,
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

    try printLabeledImage(LabeledImage{
        .label = training_labels_data.items[0],
        .image = Image{
            .pixels = training_images_data.items[0],
        },
    }, allocator);
}
