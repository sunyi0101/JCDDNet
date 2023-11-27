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
% InfoLen = 72;
% H_txt = 'Tanner.N144R12.txt';
% [LDPC_H,LDPC_M,LDPC_N,ShortenLen,PunctureLen,LDPC_row_distribution,LDPC_col_distribution,H_row_id,H_col_id_2] = ReadCode(H_txt,InfoLen,LDPC_R);
% LDPC_H1 = load('Hsys.N144R12.txt');
% LDPC_H1 = sparse(LDPC_H1);
% hEnc = comm.LDPCEncoder(LDPC_H1);

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
[bit,signal,group,~,indx0,indx1] = mapping(qAry);

%% MIMO
Nu = 1;
Ncl = 8;
Np = 10;
L = Ncl*Np;
Ntx = 2;
Nty = 2;
Nrx = 8;
Nry = 8;
Nt = Ntx * Nty;
Nr = Nrx * Nry;
Utx = 1/sqrt(Ntx)*dftmtx(Ntx);
Uty = 1/sqrt(Nty)*dftmtx(Nty);
Urx = 1/sqrt(Nrx)*dftmtx(Nrx);
Ury = 1/sqrt(Nry)*dftmtx(Nry);
Ut = zeros(Nt*Nu,Nt*Nu);
for uu = 1:Nu
    Ut(Nt*(uu-1)+1:Nt*uu,Nt*(uu-1)+1:Nt*uu) = kron(Utx,Uty);
end
Ur = kron(Urx,Ury);

if Nu == 1
    Ns = 4;
    P_base = eye(Ns);
elseif Nu == 2
    Ns = 2;
    P_base = 1/2*[1,0;0,1;1,0;0,sqrt(-1)];
else
    print('Not Supported');
end
P = zeros(Nt*Nu,Ns*Nu);
for uu = 1:Nu
    P(Nt*(uu-1)+1:Nt*uu,Ns*(uu-1)+1:Ns*uu) = P_base;
end

ML_set = FullExpansion(Ns*Nu,2^qAry,signal);
ML_ind = zeros(Ns,(2^qAry)^(Ns*Nu));
ML_bit = zeros(Ns*qAry,(2^qAry)^(Ns*Nu));
for ii = 1:(Ns*Nu)
    for jj = 1:(2^qAry)^(Ns*Nu)
        ML_ind(ii,jj) = find(ML_set(ii,jj)==signal);
        ML_bit(qAry*(ii-1)+1:qAry*ii,jj) = group(:,find(ML_set(ii,jj)==signal));
    end
end

Lp = Nt*Nu;
Ld = (LDPC_N-PunctureLen)/Ns/qAry;

Xp = zeros(Nt*Nu,Lp);
for mm = 1:Nt*Nu
    for nn = 1:Lp
        Xp(mm,nn) = exp(-2*pi*sqrt(-1)*(mm-1)*(nn-1)/max(Nt*Nu,Lp));  
    end
end

%% Simulation
icdd_flag = 1;  % 1 for ICDD; 0 for IDD
itr_max = 10;

SNR_vec = 12;
Nfrm_vec = [10000];
Nfrm = max(Nfrm_vec);

FER_cnt = zeros(Nfrm,length(SNR_vec));
FER = zeros(1,length(SNR_vec));
FER_ITR_cnt = zeros(Nfrm,length(SNR_vec));
FER_ITR = zeros(1,length(SNR_vec));

for snr = 1:length(SNR_vec)

    fprintf('qAry = %f\n',qAry);

    fprintf('SNR = %f\n',SNR_vec(snr));

    sigma2 = 10^(-SNR_vec(snr)/10)*(Ns*Nu);

    rng('default');

    for ns = 1:Nfrm_vec(snr)

        ns

        InfoBit = randi([0,1],InfoLen,Nu);
        BitStream = zeros(LDPC_N,Nu);
        for uu = 1:Nu
            BitStream(:,uu) = step(hEnc,[InfoBit(:,uu);0;0]);  % only for the case when the matrix is not full-rank
%             BitStream(:,uu) = step(hEnc,InfoBit(:,uu));
        end

        X = zeros(Ns*Nu,Ld);
        for nn = 1:Ld
            for uu = 1:Nu
                for kk = 1:Ns
                    jjj = PunctureLen + qAry*(Ns*(nn-1)+kk-1);
                    switch qAry
                        case 2
                            x_real = 1-2*BitStream(jjj+1,uu);
                            x_imag = 1-2*BitStream(jjj+2,uu);
                            X(Ns*(uu-1)+kk,nn) = sqrt(1/2)*(x_real+sqrt(-1)*x_imag);
                        case 4
                            x_real = (1-2*BitStream(jjj+1,uu))*(1+2*BitStream(jjj+3,uu));
                            x_imag = (1-2*BitStream(jjj+2,uu))*(1+2*BitStream(jjj+4,uu));
                            X(Ns*(uu-1)+kk,nn) = sqrt(1/10)*(x_real+sqrt(-1)*x_imag);
                        otherwise
                            print('not supported');
                    end
                end
            end
        end


        H = zeros(Nr,Nt*Nu);
        for uu = 1:Nu
            H(:,Nt*(uu-1)+1:Nt*uu) = mmMIMO_H(Ntx, Nty, Nrx, Nry, Ncl, Np);
        end
        noise = sqrt(sigma2/2)*(randn(Nr,Ld)+1j*randn(Nr,Ld));

        Y = H*P*X + noise;

        noise_p = sqrt(sigma2/2)*(randn(Nr,Lp)+1j*randn(Nr,Lp));
        Yp = H*Xp + noise_p;

        %% Channel Estimation
        H0 = zeros(Nr,Nt*Nu);         
        Hest_arr = PGD_H(H0,Yp,Xp,Ur,Ut,sigma2,0.01,5,100);
        Hest = Hest_arr(:,:,end);
%         MSE_INI = mean(mean(abs(Hest - H).^2))

        %% MAP+BP       
        Pr_0 = zeros(LDPC_N-PunctureLen,Nu);
        Pr_1 = zeros(LDPC_N-PunctureLen,Nu);
        log_ML_prob = zeros(Ld,(2^qAry)^(Ns*Nu));
        DemodLLR = zeros(LDPC_N-PunctureLen,Nu);
        LLR2DEC = zeros(LDPC_N,Nu);
        OutLLR = zeros(LDPC_N,Nu);
        BitEst = zeros(LDPC_N-PunctureLen,Nu);
        error = zeros(Nu,1);

        HPest = Hest*P;
        for nn = 1:Ld
            for jj = 1:(2^qAry)^(Ns*Nu)
                log_ML_prob(nn,jj) = -norm(Y(:,nn)-HPest*ML_set(:,jj))^2/sigma2;
            end
        end

        for uu = 1:Nu
            for nn = 1:Ld
                for kk = 1:Ns
                    indx = qAry*(Ns*(nn-1)+kk-1);
                    for qq = 1:size(indx0,1)
                        LLR_0 = [];
                        LLR_1 = [];
                        for ind = 1:size(indx0,2) 
                            LLR_0 = [LLR_0,log_ML_prob(nn,find(ML_ind(Ns*(uu-1)+kk,:)==indx0(qq,ind)))];
                            LLR_1 = [LLR_1,log_ML_prob(nn,find(ML_ind(Ns*(uu-1)+kk,:)==indx1(qq,ind)))];
                        end
                        DemodLLR(indx+qq,uu) = max(LLR_0) - max(LLR_1);
                    end
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

        for t = 2:itr_max
            InLLR = OutLLR(PunctureLen+1:LDPC_N,:) - DemodLLR;
            sm = zeros(Ns*Nu,Ld);
            sv = zeros(Ns*Nu,Ld);
            for nn = 1:Ld
                for uu = 1:Nu
                    for kk = 1:Ns
                        [sm(Ns*(uu-1)+kk,nn),sv(Ns*(uu-1)+kk,nn)] = cal_sm_sv(OutLLR((PunctureLen+qAry*Ns*(nn-1)+qAry*(kk-1)+1):(PunctureLen+qAry*Ns*(nn-1)+qAry*kk),uu),qAry,bit,signal);
                    end
                end
            end

            if icdd_flag == 1
                tao = 1/(norm([Xp,P*sm],2)^2);
                Hest_arr = PGD_H(H0,[Yp,Y],[Xp,P*sm],Ur,Ut,sigma2,tao/2,5,100);
                Hest = Hest_arr(:,:,end);
%                 MSE_ITR = mean(mean(abs(Hest-H).^2))
            end

            HPest = Hest*P;

            log_ML_prob = zeros(Ld,(2^qAry)^(Ns*Nu));
            for nn = 1:Ld
                for jj = 1:(2^qAry)^(Ns*Nu)
                    log_ML_prob(nn,jj) = -norm(Y(:,nn)-HPest*ML_set(:,jj))^2/sigma2;
                end
            end

            for uu = 1:Nu
                for nn = 1:Ld
                    LLR_group = InLLR((qAry*Ns*(nn-1)+1):(qAry*Ns*nn),:);
                    for kk = 1:Ns
                        indx = qAry*(Ns*(nn-1)+kk-1);
                        for qq = 1:size(indx0,1)
                            LLR_0 = [];
                            LLR_1 = [];
                            LLR_group_temp = -LLR_group;
                            LLR_group_temp(qAry*(kk-1)+qq,uu) = 0;
                            LLR_group_temp = reshape(LLR_group_temp,1,qAry*Ns*Nu);
                            for ind = 1:size(indx0,2)                                
                                bit0_group_temp = ML_bit(:,find(ML_ind(Ns*(uu-1)+kk,:)==indx0(qq,ind)));
                                ExPrior = LLR_group_temp*bit0_group_temp;
                                LLR_0 = [LLR_0,log_ML_prob(nn,find(ML_ind(Ns*(uu-1)+kk,:)==indx0(qq,ind)))+ExPrior];
                                LLR_1 = [LLR_1,log_ML_prob(nn,find(ML_ind(Ns*(uu-1)+kk,:)==indx1(qq,ind)))+ExPrior];                                
                            end
                            DemodLLR(indx+qq,uu) = max(LLR_0) - max(LLR_1);
                        end
                    end
                end
            end
    
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

    FER_ITR(snr) = sum(FER_ITR_cnt(:,snr))/Nfrm_vec(snr)/Nu;
    FER(snr) = sum(FER_cnt(:,snr))/Nfrm_vec(snr)/Nu;
 end
