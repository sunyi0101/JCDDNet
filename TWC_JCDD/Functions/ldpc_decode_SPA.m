%Gallager sum product algorithm
function [LLR_total,Lvc,iteration]=ldpc_decode_SPA(rx,H,max_iteration)
[rows,cols]=size(H);
minVal=realmin('double');
L=rx;%LLR
LLR_total = zeros(size(rx));
Lvc = repmat(rx,rows,1);
Lvc(H==0)=0;
Lcv=zeros(rows,cols);
%% check nodes
for iteration=1:max_iteration
    for j=1:rows
        cols_location=find(H(j,:)==1);
        for k=1:length(cols_location)
            temp=cols_location;
            temp(k)=[];
            w=prod(tanh(Lvc(j,temp)/2));
            if w==-1
                Lcv(j,cols_location(k))=-1/minVal;
            elseif w==1
                Lcv(j,cols_location(k))=1/minVal;
            else
                Lcv(j,cols_location(k))=2*atanh(w);
            end
        end
    end
    %% variable nodes
    for j=1:cols
        rows_location=find(H(:,j)==1);
        for k=1:length(rows_location)
            temp=rows_location;
            temp(k)=[];
            Lvc(rows_location(k),j)=L(j)+sum(Lcv(temp,j));
        end
        LLR_total(j)=L(j)+sum(Lcv(rows_location,j));
    end
    LLR_total = sign(LLR_total).*min(abs(LLR_total),10000);
    Lvc = sign(Lvc).*min(abs(Lvc),10000);
    %% early termination
    decBits=zeros(1,cols);
    decBits(LLR_total<0)=1;
    if mod(decBits*H',2)==0
        break;
    end
end