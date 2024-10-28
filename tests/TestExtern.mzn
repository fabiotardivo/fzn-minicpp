include "minicpp.mzn";
include "extern.mzn";

array [1..3] of var 1..10: x;
var int: obj;

constraint extern(x);

constraint obj = sum(x);

solve minimize obj;