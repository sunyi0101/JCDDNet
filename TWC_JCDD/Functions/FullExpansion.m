function Full = FullExpansion(P,Q,signal)

Full=zeros(P,Q^P);

for k=1:P
    for i=1:(Q^(k-1))
        for j = 1:Q
            Full(k,i+(j-1)*Q^(k-1)) = signal(j);
        end
    end           
end

for k=1:P-1
    for i=(Q^k+1):Q^P
        Full(k,i)=Full(k,i-Q^k);
    end
end

end

