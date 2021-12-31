// Coronaries bifurcation

//parameter
sten1 = 50;  // %
sten2 = 150;  // %

// local rrefinement
ref1 = 0.1;
ref2 = 0.05;
ref3 = 0.5;

// points
Point(2) = {1, 0.5, 0, ref1};
Point(3) = {0, 0, 0, ref2};
Point(5) = {0.5, 0.4, 0, ref1};
Point(6) = {2, 0.8, 0, ref3};
Point(8) = {1.5, 0.5, 0, ref1};
Point(11) = {1.9, 1, 0, ref3};
Point(12) = {1.4, 0.7, 0, ref1};
Point(13) = {0.9, 0.7-0.2*(sten1/100.0), 0, ref1};
Point(14) = {0.4, 0.6, 0, ref1};
Point(15) = {-0.1, 0.2, 0, ref1};
Point(18) = {-0.5, 0, 0, ref1};
Point(19) = {-1, -0.3, 0, ref1};
Point(22) = {-1.25, -0.3, 0, ref1};
Point(23) = {-1.5, -0.4, 0, ref3};
Point(27) = {-1.4, -0.8, 0, ref3};
Point(28) = {-1.15, -0.7, 0, ref1};
Point(29) = {-0.9, -0.7, 0, ref1};
Point(30) = {-0.4, -0.4, 0, ref1};
Point(31) = {-0.15, -0.25, 0, ref1};
Point(33) = {0.3, -0.2, 0, ref1};
Point(34) = {0.3, -0.4, 0, ref1};
Point(35) = {0.6, -0.5, 0, ref1};
Point(36) = {0.6, -0.3-0.2*(sten2/100.0), 0, ref1};
Point(37) = {0.9, -0.4, 0, ref1};
Point(38) = {1.2, -0.4, 0, ref3};
Point(39) = {1.2, -0.6, 0, ref3};
Point(40) = {0.9, -0.6, 0, ref1};


Bezier(1) = {3, 5, 2, 8, 6};
Line(2) = {6, 11};
Bezier(3) = {11, 12, 13, 14, 15};
Bezier(4) = {15, 18, 19, 22, 23};
Line(5) = {23, 27};
Bezier(6) = {27, 28, 29, 30, 31};
Bezier(7) = {31, 34, 35, 40, 39};
Line(8) = {39, 38};
Bezier(9) = {38, 37, 36, 33, 3};

Physical Curve(10) = {5};
Physical Curve(11) = {6};
Physical Curve(12) = {7};
Physical Curve(13) = {8};
Physical Curve(14) = {9};
Physical Curve(15) = {1};
Physical Curve(16) = {2};
Physical Curve(17) = {3};
Physical Curve(18) = {4};

Curve Loop(1) = {5, 6, 7, 8, 9, 1, 2, 3, 4};

Plane Surface(1) = {1};

Physical Surface(2) = {1};
