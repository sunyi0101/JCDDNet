%% proximal gradient descent
function Hest = PGD_H(H0,Y,X,Ur,Ut,sigma2,tao,miu,itr_max)
M = size(Y,1);
K = size(X,1);
Hest = zeros(M,K,itr_max);
Hvest = Ur'*H0*Ut;
for itr = 1:itr_max
    Hvest = Hvest - tao*(Hvest*Ut'*X-Ur'*Y)*(Ut'*X)'; % gradient decsent
    Hvest = Hvest./abs(Hvest).*max(abs(Hvest)-sigma2*miu*tao,zeros(size(Hvest))); % shrinkage operator
end
Hest(:,:,itr) = Ur*Hvest*Ut';
end
