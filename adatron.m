% Adatron
% X [ NxP] : P samples of dim=N
% y0 [Px1]: labels vector of P samples
% alpha [Px1] : SVM coef

function alpha = adatron( X , y0)

[N,P] = size(X);
max_epochs = 5000;
w = zeros(1,N);

max_norm = 0;
for i = 1:P
    m = norm(X(:,i),1);
    if m  > max_norm
        max_norm = m;
    end
end
eta = 0.1 * 2/max_norm;

alpha = rand(P,1);% Start with positive values for alpha
w =  (alpha' .* y0' ) * X'; % dim = [1xN]
changes = 0;
epoch = 0;

while(1)
    epoch = epoch +1;
    for i = 1:P % go through all examples
        delta = y0(i) * w * X(:,i) ;
        tmp_a = alpha(i);
        alpha(i) = alpha(i) + max([-alpha(i), eta * (1-delta)] );
        changes = changes + (alpha(i) ~= tmp_a);

        if (alpha(i) ~= tmp_a)
            % update w with the new alpha
            w =  (alpha' .* y0' ) * X'; % dim = [1xN]
        end

        % DBG
        if (sum(alpha < 0) > 0 )
            warning('alpha values<0')
        end

    end
    if (changes == 0 || epoch > max_epochs)
        break
    end

    changes = 0;

end
