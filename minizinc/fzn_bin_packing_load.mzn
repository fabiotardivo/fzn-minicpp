% Origin author: Guido Tack <tack@gecode.org>
% Reference: https://github.com/MiniZinc/libminizinc/blob/master/share/minizinc/gecode_presolver/fzn_bin_packing_load.mzn

predicate minicpp_bin_packing_load(array[int] of var int: l, array[int] of var int: bin, array[int] of int: w);

predicate fzn_bin_packing_load(array[int] of var int: l, array[int] of var int: bin, array[int] of int: w) = minicpp_bin_packing_load(l,bin,w);
