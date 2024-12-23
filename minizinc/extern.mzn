predicate minicpp_extern(array [int] of var int: x, array [int] of int: y);
predicate extern(array [int] of var int: x, array [int] of int: y) = minicpp_extern(x,y);