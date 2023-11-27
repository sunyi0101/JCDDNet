function [bit,signal,group,alpha,indx0,indx1] = mapping(qAry)
indx0 = zeros(qAry,2^(qAry-1));
indx1 = zeros(qAry,2^(qAry-1));
switch qAry
    case 2  % QPSK
        bit = [0,0,0,1,1,0,1,1];
        signal = 1/sqrt(2)*((1-2*bit(1:2:end))+sqrt(-1)*(1-2*bit(2:2:end)));
        group = reshape(bit,qAry,2^qAry);
        for qq = 1:qAry
            indx0(qq,:) = find(group(qq,:)==0);
            indx1(qq,:) = find(group(qq,:)==1);
        end
        alpha = sqrt(2);
    case 4   % 16QAM  
        bit = [0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,...
            0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,...
            1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,1,...
            1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1];
        group = reshape(bit,qAry,2^qAry);
        for qq = 1:qAry
            indx0(qq,:) = find(group(qq,:)==0);
            indx1(qq,:) = find(group(qq,:)==1);
        end
        signal = zeros(1,16);
        for pp = 1:16
            x_real = (1-2*group(1,pp))*(1+2*group(3,pp));
            x_imag = (1-2*group(2,pp))*(1+2*group(4,pp));
            signal(pp) = sqrt(1/10)*(x_real+sqrt(-1)*x_imag);
        end
        alpha = sqrt(10);
end
end

