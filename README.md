Testing whether cosine similarity reveals anything in floating-point programs

# Usage

Create a build directory and run cmake from it:

```bash
mkdir build
cd build
cmake ..
make
```

Run programs

```bash
./CG_double
./CG_float
./CG_combined
```

# Using plotter to plot X points as algorithm progresses

Use only 2 dimensional inputs for CG if you want to plot it.
```bash
./CG_double
./CG_float
cd ../CG
python3 plotter.py
```