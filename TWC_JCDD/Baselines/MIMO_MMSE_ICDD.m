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

% LDPC_R = 1/2;
% InfoLen = 144;
% H_txt = 'T.BG2.Z24.R12_N288.txt';
% [LDPC_H,LDPC_M,LDPC_N,ShortenLen,PunctureLen,LDPC_row_distribution,LDPC_col_distribution,H_row_id,H_col_id_2] = ReadCode(H_txt,InfoLen,LDPC_R);
% hEnc = comm.LDPCEncoder(LDPC_H);

ldpcPropertyValuePairs = { 'MaximumIterationCount' , 100, ...
    'ParityCheckMatrix' , LDPC_H, ...
    'DecisionMethod' , 'Soft Decision', ...
    'IterationTerminationCondition' ,'Parity check satisfied',...
    'OutputValue' , 'Whole codeword'...
    'NumIterationsOutputPort', 1};
hDec = comm.LDPCDecoder(ldpcPropertyValuePairs{:});

%% Mapping
qAry = 2;
[bit,signal,group,alpha,indx0,indx1] = mapping(qAry);

%% MIMO
Nr = 8;
Nt = 4;
Nu = 1;

cor = 0.5;
Rr = Rcor(Nr,cor);
Rt = zeros(Nt*Nu,Nt*Nu);
for uu = 1:Nu
    Rt(Nt*(uu-1)+1:Nt*uu,Nt*(uu-1)+1:Nt*uu) = Rcor(Nt,cor);
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

%% Simulation
icdd_flag = 1;  % 1 for ICDD; 0 for IDD
Turbo_max = 10;

SNR_vec = 5;
Nfrm_vec = [10000];
Nfrm = max(Nfrm_vec);

FER_cnt = zeros(Nfrm,length(SNR_vec));
FER = zeros(1,length(SNR_vec));
FER_ITR_cnt = zeros(Nfrm,length(SNR_vec));
FER_ITR = zeros(1,length(SNR_vec));

for snr = 1:length(SNR_vec)

    fprintf('qAry = %f\n',qAry);

    fprintf('SNR = %f\n',SNR_vec(snr));

    sigma2 = 10^(-SNR_vec(snr)/10)*(Nt*Nu);

    rng('default');

    for ns = 1:Nfrm_vec(snr)

        ns
        
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
                    switch qAry
                        case 2
                            x_real = 1-2*BitStream(jjj+1,uu);
                            x_imag = 1-2*BitStream(jjj+2,uu);
                            X(Nt*(uu-1)+kk,nn) = sqrt(1/2)*(x_real+sqrt(-1)*x_imag);
                        case 4
                            x_real = (1-2*BitStream(jjj+1,uu))*(1+2*BitStream(jjj+3,uu));
                            x_imag = (1-2*BitStream(jjj+2,uu))*(1+2*BitStream(jjj+4,uu));
                            X(Nt*(uu-1)+kk,nn) = sqrt(1/10)*(x_real+sqrt(-1)*x_imag);
                        otherwise
                            print('not supported');
                    end
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
        if cor == 0
            Hest = Yp*Xp'*inv(Xp*Xp'+sigma2*eye(Nt*Nu));
        else
            P = kron(transpose(Xp),eye(Nr));
            hest = inv(P'*P+sigma2*inv(cov_h))*P'*reshape(Yp,Nr*Lp,1);
            Hest = reshape(hest,Nr,Nt*Nu);
        end
%         MSE_INI = mean(mean(abs(Hest - H).^2))

        %% MMSE+BP       
        W = inv(Hest'*Hest+sigma2*eye(Nt*Nu))*Hest';
        Xest = W*Y;
        rho = real(diag(W*Hest));
        DemodLLR = zeros(LDPC_N-PunctureLen,Nu);
        LLR2DEC = zeros(LDPC_N,Nu);
        OutLLR = zeros(LDPC_N,Nu);
        BitEst = zeros(LDPC_N,Nu);
        error = ones(Nu,1);

        for uu = 1:Nu
            for kk = 1:Nt
                miu = rho(Nt*(uu-1)+kk);
                epsilon2 = miu - miu^2;
                for nn = 1:Ld
                    DemodLLR((qAry*Nt*(nn-1)+qAry*(kk-1)+1):(qAry*Nt*(nn-1)+qAry*kk),uu) = cal_LLR(Xest(Nt*(uu-1)+kk,nn),miu,epsilon2,qAry,bit,signal);
                end
            end
        end

        for uu = 1:Nu
            LLR2DEC(:,uu) = [zeros(PunctureLen,1);DemodLLR(:,uu)];
            [OutLLR(:,uu),BP] = step(hDec,LLR2DEC(:,uu));
%             [OutLLR_temp,~,BP]=ldpc_decode_SPA(LLR2DEC(:,uu).',LDPC_H,100);
%             OutLLR(:,uu) = OutLLR_temp.';
            BitEst(:,uu) = double(OutLLR(:,uu)<0);
            error(uu) = sum(BitEst(1:InfoLen,uu)~=InfoBit(:,uu));
            FER_cnt(ns,snr) = FER_cnt(ns,snr) + (error(uu)~=0);
        end

        for t = 2:Turbo_max
            InLLR = OutLLR(PunctureLen+1:LDPC_N,:) - DemodLLR;
            sm = zeros(Nt*Nu,Ld);
            sv = zeros(Nt*Nu,Ld);
            for nn = 1:Ld
                for uu = 1:Nu
                    for kk = 1:Nt
                        [sm(Nt*(uu-1)+kk,nn),sv(Nt*(uu-1)+kk,nn)] = cal_sm_sv(OutLLR((PunctureLen+qAry*Nt*(nn-1)+qAry*(kk-1)+1):(PunctureLen+qAry*Nt*(nn-1)+qAry*kk),uu),qAry,bit,signal);                        
                    end
                end
            end

            if icdd_flag == 1
                Xtemp = [Xp,sm];
                Ytemp = [Yp,Y];
                if cor == 0
                    Hest = Ytemp*Xtemp'*inv(Xtemp*Xtemp'+sigma2*eye(Nt*Nu));
                else
                    D = kron(transpose(Xtemp),eye(Nr));
                    hest = inv(D'*D+sigma2*inv(cov_h))*D'*reshape(Ytemp,Nr*(Lp+Ld),1);
                    Hest = reshape(hest,Nr,Nt);
                end
%                 MSE_ITR = mean(mean(abs(Hest-H).^2))
            end

            Xest = zeros(Nt*Nu,Ld);
            LLR = zeros(LDPC_N-PunctureLen,Nu);
            for nn = 1:Ld
                lambda = diag(sv(:,nn));
                W = Hest'*inv(Hest*lambda*Hest'+sigma2*eye(Nr));
                for uu = 1:Nu
                    for kk = 1:Nt
                        y_res = Y(:,nn) - Hest*sm(:,nn) + Hest(:,Nt*(uu-1)+kk)*sm(Nt*(uu-1)+kk,nn);
                        w_k = W(Nt*(uu-1)+kk,:);
                        Xest(Nt*(uu-1)+kk,nn) = w_k*y_res;
                        miu = real(w_k*Hest(:,Nt*(uu-1)+kk));
                        epsilon2 = miu - lambda(Nt*(uu-1)+kk)*miu^2;
                        LLR((qAry*Nt*(nn-1)+qAry*(kk-1)+1):(qAry*Nt*(nn-1)+qAry*kk),uu) = cal_LLR_SISO(Xest(Nt*(uu-1)+kk,nn),miu,epsilon2,qAry,bit,signal,InLLR((qAry*Nt*(nn-1)+qAry*(kk-1)+1):(qAry*Nt*(nn-1)+qAry*kk),uu));                
                    end
                end
            end

            DemodLLR = LLR-InLLR;
    
            for uu = 1:Nu 
                if error(uu)~= 0
                    LLR2DEC(:,uu) = [zeros(PunctureLen,1);DemodLLR(:,uu)];
                    [OutLLR(:,uu),BP] = step(hDec,LLR2DEC(:,uu));
%             [OutLLR_temp,~,BP]=ldpc_decode_SPA(LLR2DEC(:,uu).',LDPC_H,100);
%             OutLLR(:,uu) = OutLLR_temp.';
                    BitEst(:,uu) = double(OutLLR(:,uu)<0);
                    error(uu) = sum(BitEst(1:InfoLen,uu)~=InfoBit(:,uu));
                end
            end

            if sum(error) == 0
                break;
            end

        end

        FER_ITR_cnt(ns,snr) = sum(error~=0);

    end

    FER(snr) = sum(FER_cnt(:,snr))/Nfrm_vec(snr);
    FER_ITR(snr) = sum(FER_ITR_cnt(:,snr))/Nfrm_vec(snr);

end
