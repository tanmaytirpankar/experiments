Testing whether cosine similarity allows us to notice differences in 
vector directions in floating-point programs across two precisions.

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
./CG_float
./CG_double
./CG_combined_float_double
# Avoid using the above programs. They may not be entirely correct.
# They are also redundant as we have generalized programs.

./CG_mpfr
./CG_combined_hp_lp
```

# Using plotter to plot converging X points

The plotter will plot data in data_hp.csv and data_lp.csv.

cg_combined_hp_lp.cpp will automatically create these two files and populate them with data.

Change the data file names in other CG files to plot them.
Use only 2 dimensional inputs for CG if you want to plot it.
```bash
cd ../CG
python3 plotter.py
```