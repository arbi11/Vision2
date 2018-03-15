function p = lamb (m, s, x)

% i=1;
% 
% m = mu(i, :);
% s = sigma(:,:, i);
x_mu = x - m;
% a = (inv(s)*x_mu')';
% a2 = sum((x_mu).*(inv(s)*x_mu')', 2);
% a3 = exp(-0.5 * sum((x_mu).*(inv(s)*x_mu')', 2)); 
% b = det(s)^(-0.5)*(2*pi)^(-size(x, 2)/2)

p = (det(s)*((2*pi)^(-size(x, 2)/2))^(0.5)) * ...
    exp(-0.5 * sum((x_mu).*(inv(s)*x_mu')', 2));
% 
% size(sigma(:,:,1))
% 
% det(sigma(:,:,1))^(-0.5) * (2*pi)^(-size(x, 2)/2)
% size(x, 2)
% 
% (x(1:4, :) - mu(1,:))*inv(sigma(:,:,1))*(x(1:4,:) - mu(1,:))';
