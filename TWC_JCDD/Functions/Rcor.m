function R = Rcor(N,rho)
R = zeros(N,N);
for i = 1:N
    for j = 1:N
        R(i,j) = rho^abs(i-j);
    end
end
end

