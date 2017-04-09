function G = arccos_kernel(U,V)
G = 1 - (1/pi)*acos((U*V')/(sqrt(sum(U.^2,2))*sqrt(sum(V.^2,2))));
end