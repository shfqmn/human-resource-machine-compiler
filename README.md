# Human Resource Machine Compiler

A Python-to-HRM (Human Resource Machine) compiler and simulator. This tool allows you to write programs in a subset of Python and compile them into the assembly language used in the game Human Resource Machine.

## Features

- **Compiler**: Translates Python code to HRM assembly (`.hrm`).
- **Simulator**: Executes HRM programs directly from the command line.
- **Optimizer**: Performs peephole optimizations to reduce instruction count.
- **Python Subset Support**: Supports `if`, `while`, `for` loops, `input()`, `print()`, and direct memory access via a `tiles` list.

## Installation

This project requires Python 3.13 or later. No external dependencies are required.

```bash
git clone https://github.com/yourusername/human-resource-machine-compiler.git
cd human-resource-machine-compiler
```

## Usage

The main entry point is [main.py](main.py).

### Compiling a Python file

To compile a Python script into HRM assembly:

```bash
python main.py compile levels/29.py
```

This will create `levels/29.hrm`.

### Running an HRM program

To run a compiled `.hrm` program with a specific inbox:

```bash
python main.py run levels/29.hrm --inbox 1 2 3 A B
```

### Showing supported instructions

To see the list of HRM instructions supported by the assembler:

```bash
python main.py available
```

## Supported Python Subset

The compiler supports a specific subset of Python tailored for HRM's architecture:

- **Input/Output**:
  - `a = input()` translates to `INBOX`.
  - `print(a)` translates to `OUTBOX`.
- **Memory Access**:
  - Use a global `tiles` list to represent the floor tiles.
  - `tiles[0] = a` translates to `COPYTO 0`.
  - `a = tiles[0]` translates to `COPYFROM 0`.
- **Control Flow**:
  - `while True:`, `while a == b:`, etc.
  - `if a > 0:`, `if a == 0:`, etc.
  - `for i in range(start, stop, step):`
  - `try: ... except EOFError: break` for handling the end of the inbox.
- **Arithmetic**:
  - `a + b`, `a - b` (mapped to `ADD` and `SUB`).
  - `a += 1`, `a -= 1` (mapped to `BUMPUP` and `BUMPDN`).

## Configuration

You can configure the compiler by defining special variables at the top of your Python file:

- `tiles`: A list or dictionary defining the initial state and names of floor tiles.
- `allowed`: A list of strings restricting which HRM opcodes are allowed (useful for mimicking specific game levels).
- `modifiable_tiles_idx`: A list of tile indices that can be modified during execution.
- Example:

```python
allowed = ["INBOX", "OUTBOX", "COPYFROM", "COPYTO"]
tiles = ["N", "K", "A", "E", "R", "D", "O", "L", "Y", "J"]
modifiable_tiles_idx = [2, 3, 4, 5, 6, 7, 8]

def main():
    while True:
        try:
            a = int(input())
            print(tiles[a])
        except EOFError:
            break
```

## Project Structure

- [hrm/](hrm/): Core logic.
  - [compiler.py](hrm/compiler.py): Python AST to HRM assembly translator.
  - [assembler.py](hrm/assembler.py): HRM simulator and instruction definitions.
  - [optimizer.py](hrm/optimizer.py): Peephole optimizer.
- [levels/](levels/): Example programs and test cases for various HRM levels.
- [main.py](main.py): CLI entry point.
