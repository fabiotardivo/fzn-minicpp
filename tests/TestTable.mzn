include "table.mzn";
include "minicpp.mzn";

var 1..10 : x0;
var 1..10 : x1;
var 1..10 : x2;

array [int,int] of 1..10: t = [|1,2,3|3,2,1|];

constraint table([x0,x1,x2], t) ::uniud;

solve satisfy;