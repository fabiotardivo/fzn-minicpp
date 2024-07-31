predicate minicpp_table_int(array [int] of var int: x, array [int] of int: t);
predicate fzn_table_int(array [int] of var int: x, array [int, int] of int: t) = minicpp_table_int(x,array1d(t));

% Smart Tables
enum SmartTableOp = {Int, All, LtInt, LtVar, GtInt, GtVat};
predicate minicpp_smart_table_int(array [int] of var int: x, array [int] of int: t, array [int] of int: sto);
predicate smart_table(array [int] of var int: x, array [int, int] of int: t,array [int, int] of SmartTableOp: sto) = minicpp_smart_table_int(x,array1d(t),array1d(sto));

