include "minicpp.mzn";
include "extern.mzn";

array [1..3] of var 1..10: x;
var int: obj;

constraint extern(x, [3,4,5,6]);

constraint obj = sum(x);

solve minimize obj;