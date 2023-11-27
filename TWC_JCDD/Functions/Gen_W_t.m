function [W,t] = Gen_W_t(d)
t = zeros(2^(d-1),1);
W = -1*ones(2^(d-1),d);
flag_idx = 1;
for i = 1:2:d
    idx = nchoosek([1:d],i);
    t(flag_idx:flag_idx+size(idx,1)-1) = i-1;
    for jj = flag_idx:flag_idx+size(idx,1)-1
        W(jj,idx(jj-flag_idx+1,:)) = 1;
    end
    flag_idx = flag_idx+size(idx,1);
end
end

