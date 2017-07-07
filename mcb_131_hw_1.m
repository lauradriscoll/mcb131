V_set = [1 1 -1 -1; -1 1 1 -1; -1 -1 1 1; 1 -1 -1 1]';

n = 10000;
V = nan(4,n);
for x = 1:n
V(:,x) = V_set(:,randi(4));
end