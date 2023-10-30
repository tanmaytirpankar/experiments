`cg_float.cpp` and `cg_double.cpp` are the first attempts at comparing CG with float and double precision CG

`cg_combined_float_double.cpp` is the second attempt at comparing CG with float and double precision CG using 
cosine similarity

**The above programs need can be deleted as we have general versions**

`cg_mpfr.cpp` is an attempt at creating a precision generalized CG. It uses the MPFR library to create a CG.
 - Change the `prec` variable to change the precision of the CG.

`cg_combined_hp_lp.cpp` is an attempt at comparing two precision generalized CG using cosine similarity.
 - Change `hp` and `lp` variables to change the precision of the CGs.
