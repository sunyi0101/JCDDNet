function [xest,LLR_out] = GAMP(y,H,sigma2,qAry,bit,signal)
[M,K] = size(H);
LLR_out = zeros(K,qAry);
prob_out = ones(K,2^qAry);
log_prob_out = ones(K,2^qAry);

group = reshape(bit,qAry,2^qAry);

x_hat = zeros(K,1);
vx = ones(K,1);
s_hat = zeros(M,1);

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
        prob_out(kk,:) = prob_out(kk,:)./sum(prob_out(kk,:));
        x_hat(kk) = sum(signal.*prob_out(kk,:));
        vx(kk) = sum((abs(signal).^2).*prob_out(kk,:)) - abs(x_hat(kk))^2;
    end
end

xest = x_hat;

for kk = 1:K
    for qq = 1:qAry
        LLR_0 = log_prob_out(kk,find(group(qq,:)==0));
        LLR_1 = log_prob_out(kk,find(group(qq,:)==1));
        LLR_out(kk,qq) = max(LLR_0) - max(LLR_1);
    end
end

% if (sum(isnan(LLR_out)))
%     system('pause');
% end

end

