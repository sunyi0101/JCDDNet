function [ sm,sv ] = cal_sm_sv( LA,qAry,bit,signal )

group = reshape(bit,qAry,2^qAry);
prob_bit = zeros(2,qAry);
for qq = 1:qAry
    prob_bit(1,qq) = 0.5*(1+tanh(LA(qq)/2));
    prob_bit(2,qq) = 0.5*(1-tanh(LA(qq)/2));
end

prob = ones(1,2^qAry);
for pp = 1:2^qAry
    temp = group(:,pp);
    for qq = 1:qAry
        prob(pp) = prob(pp)*prob_bit(temp(qq)+1,qq);
    end
end

sm = sum(signal.*prob);
sv = sum((abs(signal).^2).*prob) - abs(sm)^2;

end

