# Deflate Compression Algorithm

This repository contains a Python implementation of the Deflate compression algorithm, which combines **LZ77** compression and **Huffman coding** to efficiently reduce file size.

## 📜 Algorithm Overview

The Deflate algorithm consists of three main stages:
1. LZ77 Compression
   - Scans the input data and replaces repeated sequences with references to previous occurrences within a sliding window (e.g., 32 KB).
   - Produces a sequence of literals (new characters) and references (offset, length pairs).
2. Huffman Encoding
   - Analyzes the frequency of literals and references from LZ77.
   - Constructs an optimal binary tree for variable-length encoding, assigning shorter codes to more frequent symbols.
3. Bitstream Formatting
   - Encodes compressed data into a properly structured bitstream.
   - Includes Huffman trees, LZ77-encoded data, and necessary markers for decompression.

## 🔧 Usage

The script uses the argparse library for command-line arguments.

### Compress a file
```python
python deflate.py -c input.txt output.deflate
```

### Decompress a file
```python
python deflate.py -d output.deflate decompressed.txt
```

### Arguments
- `-c`, `--compress` → Compress a file.
- `-d`, `--decompress` → Decompress a file.
- `input_file` → Path to the input file.
- `output_file` → Path to save the result.

## 📂 Example File Sizes

| File        | Original Size | Deflate Size | ZIP Size |
|------------|--------------|--------------|----------|
| test1.txt  | 112 KB       | 21 KB        | 6 KB     |
| test2.bin  | 253 KB       | 554 KB       | 251 KB   |

### Conclusion
- Text files compress well using Deflate.
- Binary files may increase in size if already compressed or contain high entropy data.

## 🚀 Features
- LZ77 window-based compression
- Huffman coding for entropy reduction
- Command-line support via argparse
- Bitstream formatting for compatibility
