include "table.mzn";
include "minicpp.mzn";

array [int,int] of int: t  = [|0,2,3|4,10,6|7,8,2|];
array [int,int] of SmartTableOp: sto = [|All,Int,Int|Int,GtInt,Int|Int,Int,LtVar|];

var 1..10 : x0;
var 1..10 : x1;
var 1..10 : x2;
constraint smart_table([x0,x1,x2], t, sto);

solve satisfy;