% Origin author: Guido Tack <tack@gecode.org>
% Reference: https://github.com/MiniZinc/libminizinc/blob/master/share/minizinc/gecode_presolver/fzn_bin_packing_capa.mzn

include "bin_packing_load.mzn";

predicate fzn_bin_packing_capa(array[int] of int: c, array[int] of var int: bin, array[int] of int: w) =
    let
    {
        array[min(index_set(c))..max(index_set(c))] of var 0..ub_array(c): l
    }
    in (forall( i in index_set(l) ) ( l[i] <= c[i] ) /\  bin_packing_load(l,bin,w));
