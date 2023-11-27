function [xest,LLR_out] = GAMP_SISO(x_hat,vx,y,H,sigma2,qAry,bit,signal,LLR_in)
[M,K] = size(H);
LLR_out = zeros(K,qAry);

prob_out = ones(K,2^qAry);
prob_in = ones(K,2^qAry);

log_prob_out = zeros(K,2^qAry);

group = reshape(bit,qAry,2^qAry);

s_hat = zeros(M,1);

prob_bit_in = zeros(K,qAry,2);
for kk = 1:K
    for qq = 1:qAry
        prob_bit_in(kk,qq,1) = 0.5*(1+tanh(LLR_in(kk,qq)/2));       
    end
end
prob_bit_in(:,:,2) = 1 - prob_bit_in(:,:,1);

for kk = 1:K
    for pp = 1:2^qAry
        temp = group(:,pp);        
        for qq = 1:qAry            
            prob_in(kk,pp) = prob_in(kk,pp)*prob_bit_in(kk,qq,temp(qq)+1);
        end
    end
end

for itr = 1:10
    vp = ((abs(H)).^2)*vx;
    p_hat = H*x_hat - s_hat.*vp;
    vs = 1./(vp+sigma2);
    s_hat = (y-p_hat)./(vp+sigma2);
    vtao = 1./(((abs(H)).^2).'*vs);
    tao_hat = x_hat + vtao.*(H'*s_hat);
    for kk = 1:K
        log_prob_out(kk,:) = -abs(signal-tao_hat(kk)).^2./vtao(kk);
        arg = log_prob_out(kk,:);
        prob_out(kk,:) = exp(arg-max(arg));
%         prob_out(kk,:) = -abs(signal-tao_hat(kk)).^2./vtao(kk);
        prob_out(kk,:) = prob_out(kk,:).*prob_in(kk,:);
        prob_out(kk,:) = prob_out(kk,:)./sum(prob_out(kk,:));
        x_hat(kk) = sum(signal.*prob_out(kk,:));
        vx(kk) = sum((abs(signal).^2).*prob_out(kk,:)) - abs(x_hat(kk))^2;
    end
end

xest = x_hat;

for kk = 1:K
    for qq = 1:qAry        
        log_prior_0 = -LLR_in(kk,:)*group(:,find(group(qq,:)==0));
        log_prior_1 = -LLR_in(kk,:)*group(:,find(group(qq,:)==1));
        LLR_0 = log_prob_out(kk,find(group(qq,:)==0))+log_prior_0;
        LLR_1 = log_prob_out(kk,find(group(qq,:)==1))+log_prior_1;
        LLR_out(kk,qq) = max(LLR_0) - max(LLR_1);
    end
end

% if (sum(isnan(LLR_out)))
%     system('pause');
% end

end

