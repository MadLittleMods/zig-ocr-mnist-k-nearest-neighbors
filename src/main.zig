const std = @import("std");
const bigEndianStructDeserializer = @import("big_endian_struct_deserializer.zig").bigEndianStructDeserializer;

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

const NUMBER_OF_IMAGES_TO_TRAIN_ON = 100;

const MnistImageFileHeader = extern struct {
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,

    // pub fn asBuf(self: *@This()) []u8 {
    //     return @as([*]u8, @ptrCast(self))[0..@sizeOf(@This())];
    // }
};

const Image = extern struct {
    pixels: [28 * 28]u8,
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    const file = try std.fs.cwd().openFile(TRAIN_DATA_FILE_PATH, .{});
    defer file.close();

    // Use a buffered reader for better performance (less syscalls)
    // (https://zig.news/kristoff/how-to-add-buffering-to-a-writer-reader-in-zig-7jd)
    var buffered_reader = std.io.bufferedReader(file.reader());
    var file_reader = buffered_reader.reader();

    const header = try file_reader.readStructBig(MnistImageFileHeader);
    std.log.debug("header {}", .{header});
    try std.testing.expectEqual(header.magic_number, 2051);
    try std.testing.expectEqual(header.number_of_images, 60000);
    try std.testing.expectEqual(header.number_of_rows, 28);
    try std.testing.expectEqual(header.number_of_columns, 28);

    // Make sure we don't try to allocate more images than there are in the file
    std.debug.assert(header.number_of_images >= NUMBER_OF_IMAGES_TO_TRAIN_ON);
    const image_array = try allocator.alloc(Image, NUMBER_OF_IMAGES_TO_TRAIN_ON);
    defer allocator.free(image_array);

    var deserializer = bigEndianStructDeserializer(file_reader);

    for (0..(NUMBER_OF_IMAGES_TO_TRAIN_ON - 1)) |image_index| {
        const image = try deserializer.read(Image);
        image_array[image_index] = image;
    }

    const colored_asdf = try decorateStringWithAnsiColor("asdf", 0xFF0000, allocator);
    defer allocator.free(colored_asdf);
    std.log.debug("asdf {s}", .{
        colored_asdf,
    });

    for (0..(header.number_of_rows - 1)) |image_index| {
        const start_index = image_index * header.number_of_columns;
        const end_index = start_index + header.number_of_columns;
        std.log.debug("{any}", .{image_array[0].pixels[start_index..end_index]});
    }
}
