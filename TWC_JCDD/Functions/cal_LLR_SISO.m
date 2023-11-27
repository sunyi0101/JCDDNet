function LLR = cal_LLR_SISO(est_x,miu,epsilon2,qAry,bit,signal,LA)
% max-log
group = reshape(bit,qAry,2^qAry);
LLR = zeros(1,qAry);
log_prob = zeros(1,2^qAry);
log_prior = zeros(1,2^qAry);

for pp = 1:2^qAry
    log_prob(pp) = -1/epsilon2*abs(est_x-miu*signal(pp))^2;
    log_prior(pp) = sum(-LA.*group(:,pp));
end

for qq = 1:qAry
    LLR_0 = log_prob(find(group(qq,:)==0))+log_prior(find(group(qq,:)==0));
    LLR_1 = log_prob(find(group(qq,:)==1))+log_prior(find(group(qq,:)==1));
    LLR(qq) = max(LLR_0) - max(LLR_1);
end

end

