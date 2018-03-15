clear 
img = imread('69020.jpg');
img = im2double(img);

[n, d, r] = size(img);
img = reshape(img, [n*d r]);
img = reshape(img, [n*d r]);
[n, d] = size(img);
x = img;
k =2;
eps = 0.001;
mu = img(randsample(n, k), :);
sigma = repmat(eye(d), [1, 1, k]);
r = zeros(n, k);
w = repmat(1/k, [1, k]);

LLds = [];
max_iters = 5;

while size(LLds, 1) < max_iters
    %%% E-step %%%
    for i = 1:k
        r(:,i) = w(i) * lamb (mu(i, :), sigma(:,:, i), x);
    end
    lld = sum(log(sum(r, 2)));
    
    LLds = [LLds; lld];
    r = (r./sum(r, 2));
    nks = sum(r, 1);
    %%% M-step %%%
    
    for j = 1:k
        mu(j, :)= 1./nks(j) * sum(x .* r(:,j), 1);
        x_mu = x - mu(j, :);
        sigma(:,:,j) = 1./nks(j) * ((x_mu .* r(:,j))'*x_mu);
        w(j) = 1. / n * nks(j);
    end
    
end
% 
%     if size(LLds, 1) < 2
%         continue
%     end
%     
%     if abs(lld - LLds(end - 2) < eps)
%         break
%     end
% end
% 
%     
%     
%             
    