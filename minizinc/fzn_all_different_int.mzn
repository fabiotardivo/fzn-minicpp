predicate minicpp_all_different(array[int] of var int: x);
predicate fzn_all_different_int(array[int] of var int: x) = minicpp_all_different(x);
