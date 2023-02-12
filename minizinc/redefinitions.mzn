% Reification
predicate bool_and_reif(var bool: a, var bool: b, var bool: r);
predicate bool_and(var bool: a, var bool: b, var bool: r) = bool_and_reif(a,b,r);
predicate bool_or_reif(var bool: a, var bool: b, var bool: r);
predicate bool_or(var bool: a, var bool: b, var bool: r) = bool_or_reif(a,b,r);
predicate bool_xor_reif(var bool: a, var bool: b, var bool: r);
predicate bool_xor(var bool: a, var bool: b, var bool: r) = bool_xor_reif(a,b,r);
predicate array_bool_and_reif(array [int] of var bool: as, var bool: r);
predicate array_bool_and(array [int] of var bool: as, var bool: r) = array_bool_and_reif(as,r);
predicate array_bool_or_reif(array [int] of var bool: as, var bool: r);
predicate array_bool_or(array [int] of var bool: as, var bool: r) = array_bool_or_reif(as,r);

% Implications
predicate int_eq_imp(var int: a, var int: b, var bool: r);
predicate int_ne_imp(var int: a, var int: b, var bool: r);
predicate int_le_imp(var int: a, var int: b, var bool: r);
predicate int_lt_imp(var int: a, var int: b, var bool: r);
predicate int_lin_eq_imp(array [int] of int: as, array [int] of var int: bs, int: c, var bool: r);
predicate int_lin_ne_imp(array [int] of int: as, array [int] of var int: bs, int: c, var bool: r);
predicate int_lin_le_imp(array [int] of int: as, array [int] of var int: bs, int: c, var bool: r);
predicate array_bool_and_imp(array [int] of var bool: as, var bool: r);
predicate array_bool_or_imp(array [int] of var bool: as, var bool: r);
predicate array_bool_xor_imp(array [int] of var bool: as, var bool: r);
predicate bool_and_imp(var bool: a, var bool: b, var bool: r);
predicate bool_or_imp(var bool: a, var bool: b, var bool: r);
predicate bool_xor_imp(var bool: a, var bool: b, var bool: r);
predicate bool_eq_imp(var bool: a, var bool: b, var bool: r);
predicate bool_le_imp(var bool: a, var bool: b, var bool: r);
predicate bool_lt_imp(var bool: a, var bool: b, var bool: r);
