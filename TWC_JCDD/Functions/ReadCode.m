function [LDPC_H,LDPC_M,LDPC_N,ShortenLen,PunctureLen,LDPC_row_distribution,LDPC_col_distribution,H_row_id,H_col_id_2] = ReadCode(H_txt,InfoLen,LDPC_R)
fid = fopen(char(H_txt));
line = str2num(fgetl(fid));
LDPC_N = line(1);                                                   
LDPC_M = line(2);                                                   
LDPC_K = LDPC_N-LDPC_M;                                             
ShortenLen = LDPC_K - InfoLen;                                      
PunctureLen = LDPC_N - ShortenLen - (LDPC_K - ShortenLen)/LDPC_R;   
LDPC_col_distribution = zeros(LDPC_N,1);
LDPC_row_distribution = zeros(LDPC_M,1);
fgetl(fid);fgetl(fid);fgetl(fid);
H_row_id = [];
H_col_id = [];
for col = 1:LDPC_N
    row_id = str2num(fgetl(fid));
    row_id(row_id == 0) = [];
    H_row_id = [H_row_id, row_id];
    LDPC_col_distribution(col)=length(row_id);
    col_id = col * ones(1,length(row_id));
    H_col_id=[H_col_id, col_id];
end
H_row_id_2 = [];
H_col_id_2= [];
for row = 1:LDPC_M
    col_id = str2num(fgetl(fid));
    col_id(col_id == 0) = [];
    H_col_id_2 = [H_col_id_2, col_id];
    LDPC_row_distribution(row)=length(col_id);
    row_id = row * ones(1,length(col_id));
    H_row_id_2=[H_row_id_2, row_id];
end
fclose(fid);
LDPC_H = sparse(H_row_id, H_col_id, ones(1,length(H_row_id)));
LDPC_H = (LDPC_H >= 1);
end

