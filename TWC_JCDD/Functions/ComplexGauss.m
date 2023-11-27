function y = ComplexGauss(x,miu,sigma2)
y = 1/(pi*sigma2).*exp(-abs(x-miu).^2./sigma2);
end

