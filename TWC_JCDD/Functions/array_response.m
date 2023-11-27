function y = array_response(phi,theta,M,N)
y = zeros(1,M*N);
for m= 0:M-1
    for n= 0:N-1
        y(m*N+n+1) = exp( 1i* pi* ( m*sin(phi)*sin(theta) + n*cos(theta) ) );
    end
end
y = y.'/sqrt(M*N);
end

