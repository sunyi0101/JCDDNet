function [A,theta,lambda_A,N_neq] = Gen_A_theta(LDPC_M,LDPC_N,LDPC_row_distribution,H_col_id_2)
A = [];
theta = [];
N_neq = 0;
for jj = 1:LDPC_M
    N_neq = N_neq + 2^(LDPC_row_distribution(jj)-1);
end
for jj = 1:LDPC_M
    [W_jj,t_jj] = Gen_W_t(LDPC_row_distribution(jj));
    theta = [theta;t_jj];
    ind_ii = sum(LDPC_row_distribution(1:jj-1))+1:sum(LDPC_row_distribution(1:jj)); 
    v_ii = H_col_id_2(ind_ii);
    Q_jj = zeros(LDPC_row_distribution(jj),LDPC_N);
    for kk = 1:LDPC_row_distribution(jj)
        Q_jj(kk,v_ii(kk)) = 1;
    end
    A=[A;W_jj*Q_jj];
end
lambda_A = diag(A'*A);
end

