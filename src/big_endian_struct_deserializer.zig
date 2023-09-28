const std = @import("std");

const bigToNative = std.mem.bigToNative;

// via https://nathancraddock.com/blog/2022/deserialization-with-zig-metaprogramming/#writing-a-struct-deserializer-with-zig
pub const BigEndianStructDeserializer = struct {
    bytes: []const u8,
    index: usize,

    pub fn init(bytes: []const u8) @This() {
        return .{ .bytes = bytes, .index = 0 };
    }

    fn readInt(self: *@This(), comptime T: type) !T {
        const size = @sizeOf(T);
        if (self.index + size > self.bytes.len) return error.EndOfStream;

        const slice = self.bytes[self.index .. self.index + size];
        const value = @as(*align(1) const T, @ptrCast(slice)).*;

        self.index += size;
        return bigToNative(T, value);
    }

    fn readFloat(self: *@This()) !f32 {
        const size = @sizeOf(f32);
        if (self.index + size > self.bytes.len) return error.EndOfStream;

        const slice = self.bytes[self.index .. self.index + size];
        const value = @as(*align(1) const u32, @ptrCast(slice)).*;

        self.index += size;
        return @as(f32, @bitCast(bigToNative(u32, value)));
    }

    fn readStruct(self: *@This(), comptime T: type) !T {
        const fields = std.meta.fields(T);

        var item: T = undefined;
        inline for (fields) |field| {
            @field(item, field.name) = try self.read(field.type);
        }

        return item;
    }

    pub fn read(self: *@This(), comptime T: type) !T {
        return switch (@typeInfo(T)) {
            .Int => try self.readInt(T),
            .Float => try self.readFloat(),
            .Array => |array| {
                var arr: [array.len]array.child = undefined;
                var index: usize = 0;
                while (index < array.len) : (index += 1) {
                    arr[index] = try self.read(array.child);
                }
                return arr;
            },
            .Struct => try self.readStruct(T),
            else => @compileError("unsupported type"),
        };
    }
};
