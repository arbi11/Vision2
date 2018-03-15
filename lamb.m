function p = lamb (m, s, x)


x_mu = x - m;

p = det(s)^(-0.5)^(2*pi)^(-size(x, 1)/2) * ...
    exp(-0.5 * sum((x_mu).*(inv(s)*x_mu')', 2));
% 
% size(sigma(:,:,1))
% 
% det(sigma(:,:,1))^(-0.5) * (2*pi)^(-size(x, 2)/2)
% size(x, 2)
% 
% (x(1:4, :) - mu(1,:))*inv(sigma(:,:,1))*(x(1:4,:) - mu(1,:))';
