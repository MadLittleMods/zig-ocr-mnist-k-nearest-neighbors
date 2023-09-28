const std = @import("std");

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

const MnistImageFileHeader = extern struct {
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,

    pub fn asBuf(self: *@This()) []u8 {
        return @as([*]u8, @ptrCast(self))[0..@sizeOf(@This())];
    }
};

// const Image = extern struct {
//     [28 * 28]u8;
// }

pub fn main() !void {
    const file = try std.fs.cwd().openFile(TRAIN_DATA_FILE_PATH, .{});
    defer file.close();
    //try file.seekTo(0);

    var buffered_reader = std.io.bufferedReader(file.reader());
    var file_reader = buffered_reader.reader();

    const header = try file_reader.readStructBig(MnistImageFileHeader);
    std.log.debug("header {}", .{header});

    try std.testing.expectEqual(header.magic_number, 2051);
    try std.testing.expectEqual(header.number_of_images, 60000);
    try std.testing.expectEqual(header.number_of_rows, 28);
    try std.testing.expectEqual(header.number_of_columns, 28);
}
