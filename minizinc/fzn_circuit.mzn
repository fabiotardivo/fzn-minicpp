predicate minicpp_circuit(array[int] of var int: x);
predicate fzn_circuit(array[int] of var int: x) = minicpp_circuit(x);
