# Data Structures

Implementations from scratch, no standard library containers. Personal reference.

## Implementations

| Data Structure | C++ | Python | Rust | Go |
|----------------|-----|--------|------|-----|
| Static Array | ✓ | ✓ | ✓ | ✓ |
| Dynamic Array | ✓ | ✓ | ✓ | ✓ |

## C++

Requires CMake and a C++17 compiler.

```bash
cd data-structures/cpp
cmake -B build
cmake --build build
./build/static_array_test
./build/dynamic_array_test
```

## Python

Requires Python 3. No external dependencies.

```bash
cd data-structures/python
python3 -m unittest discover tests -v
```

## Rust

Requires Rust (cargo).

```bash
cd data-structures/rust
cargo test
```

## Go

Requires Go 1.21+.

```bash
cd data-structures/go
go test -v ./...
```
