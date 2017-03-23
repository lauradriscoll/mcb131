
function[w, converged, epochs, error_history] = perceptron(X,y0)
[N, P] = size(X);
tmax = 5000;
converged = 0;
epochs = 0;

max_norm = 0;
for i = 1:P
    m = norm(X(:,i),1);
    if m  > max_norm
        max_norm = m;
    end
end
eta = 0.1 * 2/max_norm;

conv_ind = zeros(size(y0));
w_history = nan(tmax,N);
error_history = nan(tmax,1);
w = rand(size(X,1),1);

while converged==0 && epochs<tmax
    epochs = epochs+1;
for i = 1:P
    if w'*X(:,i)*y0(i)<0
        
    w = w+eta*X(:,i)*y0(i);
    
    conv_ind(i) = 0;
    
    else
    conv_ind(i) = 1;
    end
end

converged = isequal(sum(conv_ind),P);
error_history(epochs) = sum(conv_ind==1);
w_history(epochs,:) = w;

end
end