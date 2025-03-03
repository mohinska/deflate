"""
Lossless data compression file format that uses a combination of LZ77 and Huffman coding.
"""

import struct
import argparse
import heapq

SPECIAL_EOF = 256

class LZ77:
    """Implements the LZ77 compression algorithm."""

    def __init__(self, window_size=32000):
        """Initialize the LZ77 compressor with a specified window size."""
        self.window_size = window_size

    def find_longest_match(self, data: bytes, pos: int):
        """Find the longest match of the upcoming data in the sliding window."""
        best_length = 0
        best_offset = 0
        start_index = max(0, pos - self.window_size)

        for i in range(start_index, pos):
            length = 0

            while pos + length < len(data) and data[i + length] == data[pos + length]:
                length += 1

            if length > best_length:
                best_length = length
                best_offset = pos - i
        return best_offset, best_length

    def compress(self, data: bytes) -> list:
        """Compress the given data using LZ77 and return a list of tokens."""
        tokens = []
        pos = 0
        while pos < len(data):
            best_offset, best_length = self.find_longest_match(data, pos)

            if pos + best_length < len(data):
                next_char = data[pos + best_length]
            else:
                next_char = None
            tokens.append((best_offset, best_length, next_char))

            pos += best_length + 1
        return tokens

    def decompress(self, tokens: list) -> bytearray:
        """Decompress a list of LZ77 tokens back into the original data."""
        output = bytearray()
        for offset, length, next_char in tokens:
            if offset == 0 and length == 0:

                if next_char is not None:
                    output.append(next_char)
            else:
                start = len(output) - offset
                for i in range(length):
                    output.append(output[start + i])

                if next_char is not None:
                    output.append(next_char)
        return output


class Huffman:
    """Implements Huffman coding for compression."""

    class HuffmanNode:
        """Represents a node in the Huffman tree."""
        def __init__(self, symbol=None, freq=0, left=None, right=None):
            self.symbol = symbol
            self.freq = freq
            self.left = left
            self.right = right

    @staticmethod
    def build_tree(freq_table: dict):
        """Build a Huffman tree from a dictionary mapping symbols to frequencies."""
        heap = []
        count = 0
        for symbol, freq in freq_table.items():
            node = Huffman.HuffmanNode(symbol, freq)
            heapq.heappush(heap, (freq, count, node))
            count += 1
        if not heap:
            return None
        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            merged = Huffman.HuffmanNode(None, freq1 + freq2, node1, node2)
            heapq.heappush(heap, (merged.freq, count, merged))
            count += 1
        return heap[0][2]

    @staticmethod
    def build_codes(node, prefix="", table=None):
        """Recursively traverse the Huffman tree to build a
        dictionary mapping symbols to bit strings."""
        if table is None:
            table = {}
        if node.symbol is not None:
            table[node.symbol] = prefix if prefix != "" else "0"
        else:
            Huffman.build_codes(node.left, prefix + "0", table)
            Huffman.build_codes(node.right, prefix + "1", table)
        return table


def bits_to_bytes(bit_str: str) -> bytes:
    """Convert a bit string into bytes, padding with zeros if necessary."""
    padding = (8 - len(bit_str) % 8) % 8
    bit_str += "0" * padding
    b = bytearray()
    for i in range(0, len(bit_str), 8):
        byte = int(bit_str[i:i+8], 2)
        b.append(byte)
    return bytes(b)


class BitReader:
    """Helper class to read individual bits from a bytes object."""

    def __init__(self, data: bytes):
        self.data = data
        self.bit_pos = 0

    def read_bit(self):
        """Read a single bit from the data."""
        byte_index = self.bit_pos // 8
        if byte_index >= len(self.data):
            return None
        bit_index = 7 - (self.bit_pos % 8)
        self.bit_pos += 1
        return (self.data[byte_index] >> bit_index) & 1


def deflate_compress(input_file_path: str, output_file_path: str):
    """Compress a file using the simplified Deflate algorithm."""
    with open(input_file_path, "rb") as f:
        data = f.read()

    tokens = LZ77().compress(data)

    flat_symbols = []
    for offset, length, next_char in tokens:
        flat_symbols.append(offset)
        flat_symbols.append(length)
        flat_symbols.append(
            next_char if next_char is not None else SPECIAL_EOF)
    total_symbols = len(flat_symbols)

    freq_table = {}
    for sym in flat_symbols:
        freq_table[sym] = freq_table.get(sym, 0) + 1

    huff_tree = Huffman.build_tree(freq_table)
    code_table = Huffman.build_codes(huff_tree)

    bit_str = ""
    for sym in flat_symbols:
        bit_str += code_table[sym]
    encoded_bytes = bits_to_bytes(bit_str)

    header = bytearray()
    header += struct.pack("<I", total_symbols)
    unique_count = len(freq_table)
    header += struct.pack("<I", unique_count)
    for sym, freq in freq_table.items():
        header += struct.pack("<I", sym)
        header += struct.pack("<I", freq)

    with open(output_file_path, "wb") as f:
        f.write(header)
        f.write(encoded_bytes)
    print(f"Compression complete: {input_file_path} -> {output_file_path}")


def deflate_decompress(input_file_path: str, output_file_path: str):
    """Decompress a file compressed with the simplified Deflate algorithm."""
    with open(input_file_path, "rb") as f:
        file_content = f.read()

    pos = 0
    total_symbols, = struct.unpack("<I", file_content[pos:pos+4])
    pos += 4
    unique_count, = struct.unpack("<I", file_content[pos:pos+4])
    pos += 4
    freq_table = {}
    for _ in range(unique_count):
        sym, = struct.unpack("<I", file_content[pos:pos+4])
        pos += 4
        freq, = struct.unpack("<I", file_content[pos:pos+4])
        pos += 4
        freq_table[sym] = freq

    huff_tree = Huffman.build_tree(freq_table)

    encoded_bytes = file_content[pos:]
    bit_reader = BitReader(encoded_bytes)

    flat_symbols = []
    node = huff_tree
    while len(flat_symbols) < total_symbols:
        bit = bit_reader.read_bit()
        if bit is None:
            break
        if bit == 0:
            node = node.left
        else:
            node = node.right
        if node.symbol is not None:
            flat_symbols.append(node.symbol)
            node = huff_tree

    if len(flat_symbols) != total_symbols:
        raise ValueError("Decoded symbol count does not match header info.")

    tokens = []
    for i in range(0, len(flat_symbols), 3):
        offset = flat_symbols[i]
        length = flat_symbols[i+1]
        next_char = flat_symbols[i+2]
        if next_char == SPECIAL_EOF:
            next_char = None
        tokens.append((offset, length, next_char))

    original_data = LZ77().decompress(tokens)

    with open(output_file_path, "wb") as f:
        f.write(original_data)
    print(f"Decompression complete: {input_file_path} -> {output_file_path}")


def main():
    """Main function to handle command-line arguments for compression or decompression."""
    parser = argparse.ArgumentParser(
        description="Simplified Deflate Compressor/Decompressor with \
LZ77 triple tokens (offset, length, next_char)")
    parser.add_argument("mode", choices=["compress", "decompress"],
                        help="Mode: compress or decompress")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    args = parser.parse_args()

    if args.mode == "compress":
        deflate_compress(args.input, args.output)
    else:
        deflate_decompress(args.input, args.output)


if __name__ == "__main__":
    main()
