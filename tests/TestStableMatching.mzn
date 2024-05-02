include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 3;
var 1..n : m0;
var 1..n : m1;
var 1..n : m2;

var 1..n : w0;
var 1..n : w1;
var 1..n : w2;

array [int,int] of 1..n: pm = [|1,2,3|3,2,1|2,3,1|];
array [int,int] of 1..n: pw = [|1,2,3|3,2,1|2,3,1|];

constraint stable_matching([m0,m1,m2], [w0,w1,w2], pm, pw) ::uniud;

solve satisfy;