function H = mmMIMO_H(Ntx, Nty, Nrx, Nry, Ncl, Np)
L = Ncl*Np;
At = zeros(L,2);
Ar = zeros(L,2);
for cl=1:Ncl

    At(Np*(cl-1)+1,:) = pi*rand(1, 2) - pi/2;  %-2/pi~2/pi
    Ar(Np*(cl-1)+1,:) = pi*rand(1, 2) - pi/2;  %-2/pi~2/pi
    for p = 2:L/Ncl
        At(Np*(cl-1)+p,:) = At(Np*(cl-1)+1,:) + pi/24*rand(1, 2) - pi/48;
        Ar(Np*(cl-1)+p,:) = Ar(Np*(cl-1)+1,:) + pi/24*rand(1, 2) - pi/48;
    end

end

alpha(1) = (randn(1,1)+1i*randn(1,1))/sqrt(2); % gain of the LoS
alpha(2:L) = 10^(-0.5)*(randn(1,L-1)+1i*randn(1,L-1))/sqrt(2); 

Nt = Ntx * Nty;
Nr = Nrx * Nry;
H = zeros(Nr,Nt);
for l=1:L
    ar = array_response(Ar(l,1),Ar(l,2), Nrx, Nry);
    at = array_response(At(l,1),At(l,2), Ntx, Nty);
    H = H + sqrt(Nr * Nt / L)*alpha(l)*ar*at';
end
end