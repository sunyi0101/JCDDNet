warning('off');
clear all;
clc;

%% LDPC
LDPC_R = 1/2;
InfoLen = 144;
H_txt = 'Tanner.dv3.dc6.N288.txt';
[LDPC_H,LDPC_M,LDPC_N,ShortenLen,PunctureLen,LDPC_row_distribution,LDPC_col_distribution,H_row_id,H_col_id_2] = ReadCode(H_txt,InfoLen,LDPC_R);
LDPC_H1 = load('H_sys_N288.txt');
LDPC_H1 = sparse(LDPC_H1);
hEnc = comm.LDPCEncoder(LDPC_H1);

[A,theta,lambda_A,N_neq] = Gen_A_theta(LDPC_M,LDPC_N,LDPC_row_distribution,H_col_id_2);

%% Mapping
qAry = 4;
[bit,signal,group,~,indx0,indx1] = mapping(qAry);

%% MIMO
Nr = 8;
Nt = 4;
Nu = 1;

ML_set = FullExpansion(Nt,2^qAry,signal);
ML_ind = zeros(Nt,(2^qAry)^Nt);
ML_bit = zeros(Nt*qAry,(2^qAry)^Nt);
for ii = 1:Nt
    for jj = 1:(2^qAry)^Nt
        ML_ind(ii,jj) = find(ML_set(ii,jj)==signal);
        ML_bit(qAry*(ii-1)+1:qAry*ii,jj) = group(:,find(ML_set(ii,jj)==signal));
    end
end

Hcor = 0;
Rr = Rcor(Nr,Hcor);
Rt = zeros(Nt*Nu,Nt*Nu);
for uu = 1:Nu
    Rt(Nt*(uu-1)+1:Nt*uu,Nt*(uu-1)+1:Nt*uu) = Rcor(Nt,Hcor);
end
cov_h = kron(Rt,Rr);

Lp = Nt*Nu;
Ld = (LDPC_N-PunctureLen)/Nt/qAry;

Xp = zeros(Nt*Nu,Lp);
for mm = 1:Nt*Nu
    for nn = 1:Lp
        Xp(mm,nn) = exp(-2*pi*sqrt(-1)*(mm-1)*(nn-1)/max(Nt*Nu,Lp)); 
    end
end

%% load paramters optimized via offline training
if Hcor == 0
    load('Trained_16QAM_iid_R8T4.mat');
else
    load('Trained_16QAM_cor_R8T4.mat');
end

%% Simulation
ADMM_ITR = 500;
SNR_vec = 8;
Nfrm_vec = [10000];
Nfrm = max(Nfrm_vec);

FER_500_cnt = zeros(Nfrm,length(SNR_vec));
FER_500 = zeros(1,length(SNR_vec));
FER_100_cnt = zeros(Nfrm,length(SNR_vec));
FER_100 = zeros(1,length(SNR_vec));
FER_200_cnt = zeros(Nfrm,length(SNR_vec));
FER_200 = zeros(1,length(SNR_vec));
ADMM_cnt = zeros(Nfrm,length(SNR_vec));
ADMM_ave =  zeros(1,length(SNR_vec));

for snr = 1:length(SNR_vec)

    fprintf('qAry = %f\n',qAry);

    fprintf('SNR = %f\n',SNR_vec(snr));

    sigma2 = 10^(-SNR_vec(snr)/10)*(Nt*Nu);

    rng('default');

    for ns = 1:Nfrm_vec(snr)

        InfoBit = randi([0,1],InfoLen,Nu);
        BitStream = zeros(LDPC_N,Nu);
        for uu = 1:Nu
            BitStream(:,uu) = step(hEnc,[InfoBit(:,uu);0;0]);  % only for the case when the matrix is not full-rank
%             BitStream(:,uu) = step(hEnc,InfoBit(:,uu));
        end

        X = zeros(Nt*Nu,Ld);
        for nn = 1:Ld
            for uu = 1:Nu
                for kk = 1:Nt
                    jjj = PunctureLen + qAry*(Nt*(nn-1)+kk-1);
                    x_real = (1-2*BitStream(jjj+1,uu))*(1+2*BitStream(jjj+3,uu));
                    x_imag = (1-2*BitStream(jjj+2,uu))*(1+2*BitStream(jjj+4,uu));
                    X(Nt*(uu-1)+kk,nn) = sqrt(1/10)*(x_real+sqrt(-1)*x_imag);
                end
            end
        end

        H = sqrt(1/2)*(randn(Nr,Nt*Nu)+1j*randn(Nr,Nt*Nu));
        H = (Rr^0.5)*H*(Rt^0.5);
        noise = sqrt(sigma2/2)*(randn(Nr,Ld)+1j*randn(Nr,Ld));

        Y = H*X + noise;

        noise_p = sqrt(sigma2/2)*(randn(Nr,Lp)+1j*randn(Nr,Lp));
        Yp = H*Xp + noise_p;

        %% Channel Estimation     
        if Hcor == 0
            Hest = Yp*Xp'*inv(Xp*Xp'+sigma2*eye(Nt*Nu));
        else
            P = kron(transpose(Xp),eye(Nr));
            hest = inv(P'*P+sigma2*inv(cov_h))*P'*reshape(Yp,Nr*Lp,1);
            Hest = reshape(hest,Nr,Nt*Nu);
        end
%         MSE_INI = mean(mean(abs(Hest - H).^2))

        %% MMSE     
        W = inv(Hest'*Hest+sigma2*eye(Nt*Nu))*Hest';
        Xest = W*Y;
        rho = real(diag(W*Hest));
        DemodLLR = zeros(LDPC_N-PunctureLen,Nu);        
        BitEst = 1/2*ones(LDPC_N,Nu);
        error = zeros(Nu,1);

        for uu = 1:Nu
            for kk = 1:Nt
                miu_k = rho(Nt*(uu-1)+kk);
                epsilon2_k = miu_k - miu_k^2;
                for nn = 1:Ld
                    DemodLLR((qAry*Nt*(nn-1)+qAry*(kk-1)+1):(qAry*Nt*(nn-1)+qAry*kk),uu) = cal_LLR(Xest(Nt*(uu-1)+kk,nn),miu_k,epsilon2_k,qAry,bit,signal);
                end
            end            
        end

        for uu = 1:Nu
            BitEst(PunctureLen+1:LDPC_N,uu) = double(DemodLLR(:,uu)<0);
            error(uu) = sum(BitEst(:,uu)~=BitStream(:,uu));
        end

        u = 1/2*ones(LDPC_N,Nu);       
        for nn = 1:Ld
            for uu = 1:Nu
                for kk = 1:Nt
                    jj = qAry*(Nt*(nn-1)+kk-1);
                    for ii = (jj+1):(jj+qAry)                        
                        u(PunctureLen + ii,uu) = 1/(1+exp(DemodLLR(ii,uu)));
                        if u(PunctureLen + ii,uu) > 1
                            u(PunctureLen + ii,uu) = 1;
                        elseif u(PunctureLen + ii,uu) <0
                            u(PunctureLen + ii,uu) = 0;
                        end
                    end
                end
            end
        end
 
        %% ADMM
       
        lambda = zeros(size(A,1),Nu);
        z = zeros(size(lambda));
        
        X_temp = Xest;
        XX = [X_temp,Xp];
        YY = [Y,Yp];
        W_temp = inv(XX*XX'+sigma2*1/max(eig(cov_h))*eye(Nt*Nu))*XX*YY';
        lambda_max_temp = max(eig(W_temp*W_temp'));
        F_temp = W_temp*Y + lambda_max_temp*X_temp - W_temp*W_temp'*X_temp;
        F_temp_real = [real(F_temp);imag(F_temp)];

        for itr = 1:ADMM_ITR

            miu = miu_vec(itr);
            alpha1 = alpha1_vec(itr);
            alpha2 = alpha2_vec(itr);
            factor = factor_vec(itr);
            if Hcor == 0
                cor = 1;
            else
                cor = cor_vec(itr);
            end
            OR = OR_vec(itr);
            acc = acc_vec(itr);

            z_old = z;
            lambda_old = lambda;

            indu = reshape([PunctureLen+1:LDPC_N],4,(LDPC_N-PunctureLen)/4);
            indu1 = reshape(indu(1:2,:),(LDPC_N-PunctureLen)/2,1);
            indu2 = reshape(indu(3:4,:),(LDPC_N-PunctureLen)/2,1);
            res = reshape(F_temp_real,Nt*Nu,2,Ld);            

            %% step 1
            for uu = 1:Nu
                if error(uu) ~= 0
                    u21 = u(indu1,uu);
                    u22 = u(indu2,uu);

                    res_u = res(Nt*(uu-1)+1:Nt*uu,:,:);
                    res2 = reshape(permute(res_u,[2,1,3]),Nt*Ld*2,1);

                    fb1 = 1/10*8*lambda_max_temp*(1+2*u22).^2;
                    fb_res1 = 1/10*4*lambda_max_temp*(1+2*u22).^2 - sqrt(1/10)*4*(1+2*u22).*res2;
                    q1 = miu*A(:,indu1)'*(theta-z(:,uu)-lambda(:,uu)) + fb_res1 - alpha1;
                    w1 = miu*lambda_A(indu1) + fb1 - 2*alpha1;
                    u21 = q1./w1;
                    u21 = max(u21,0) - max(u21-1,0);

                    fb2 = 1/10*8*lambda_max_temp*(1-2*u21).^2;
                    fb_res2 = -1/10*4*lambda_max_temp*(1-2*u21).^2 + sqrt(1/10)*4*(1-2*u21).*res2;
                    q2 = miu*A(:,indu2)'*(theta-z(:,uu)-lambda(:,uu)) + fb_res2 - alpha2;
                    w2 = miu*lambda_A(indu2) + fb2 - 2*alpha2;
                    u22 = q2./w2;
                    u22 = max(u22,0) - max(u22-1,0);

                    utemp = [reshape(u21,2,(LDPC_N-PunctureLen)/4);reshape(u22,2,(LDPC_N-PunctureLen)/4)];
                    u(:,uu) = [reshape(utemp,LDPC_N-PunctureLen,1)];

%                     qp = miu*A(:,1:PunctureLen)'*(theta-z-lambda) - alpha2;
%                     wp = miu*lambda_A(1:PunctureLen) - 2*alpha2;
%                     up = qp./wp;
%                     up = max(up,0) - max(up-1,0);
% 
%                     u(:,uu) = [up;reshape(utemp,LDPC_N-PunctureLen,1)];
                end
            end

            for uu = 1:Nu
                bit_sec = reshape(u(PunctureLen+1:LDPC_N,uu),qAry,Nt,Ld);
                real_X_temp = squeeze((1 - 2*bit_sec(1,:,:)).*(1 + 2*bit_sec(3,:,:)));
                imag_X_temp = squeeze((1 - 2*bit_sec(2,:,:)).*(1 + 2*bit_sec(4,:,:)));
                X_temp((Nt*(uu-1)+1):Nt*uu,:) = sqrt(1/10)*(real_X_temp + sqrt(-1)*imag_X_temp);
            end
    
            W_temp = inv([X_temp,Xp]*[X_temp,Xp]'+sigma2*cor*eye(Nt*Nu))*[X_temp,Xp]*[Y,Yp]';
            lambda_max_temp = factor*lambda_max_temp;
            F_temp = W_temp*Y + lambda_max_temp*X_temp - W_temp*W_temp'*X_temp;
            F_temp_real = [real(F_temp);imag(F_temp)];

            %% step 2
            ztemp = theta - OR*A*u - (1-OR)*(theta - z_old) - lambda;
            z = ztemp;
            z(z<0) = 0;

            %% step 3
            lambda = -ztemp + z ;

            %% acc
            z = z + acc*(z-z_old);
            lambda = lambda + acc*(lambda-lambda_old);

            for uu = 1:Nu
                BitEst(:,uu) = double(u(:,uu)>=0.5);
                error(uu) = sum(BitEst(:,uu)~=BitStream(:,uu));
                if error(uu) == 0
                    u(:,uu) = BitEst(:,uu);
                end
            end

            if itr <= 100
                FER_100_cnt(ns,snr) = sum(error~=0);
            end

            if itr <= 200
                FER_200_cnt(ns,snr) = sum(error~=0);
            end

            if mod(LDPC_H*BitEst,2)==zeros(LDPC_M,Nu)
                if sum(sum(BitEst))~=0 && sum(sum(BitEst))~= Nu*LDPC_N
                    break;
                end
            end

        end
        
        fprintf('itr = %f\n',itr);
        ADMM_cnt(ns,snr) = itr;


        FER_500_cnt(ns,snr) = sum(error~=0);

    end

    FER_100(snr) = sum(FER_100_cnt(:,snr))/Nfrm_vec(snr);
    FER_200(snr) = sum(FER_200_cnt(:,snr))/Nfrm_vec(snr);
    FER_500(snr) = sum(FER_500_cnt(:,snr))/Nfrm_vec(snr);
    ADMM_ave(snr) = sum(ADMM_cnt(:,snr))/Nfrm_vec(snr);
end
