close all
%% 2 a
tau = .1;
v0 = .1;
a_set = [-2 2.001 4];
tmax = 100;
x = nan(size(a_set,2),tmax);
y = nan(size(a_set,2),tmax);

for a=a_set
b = tau*v0/(a-2);
for t = 1:tmax
x(a==a_set,t) = -b*(-1+exp((2-a)*t/tau));
y(a==a_set,t) = -b*(-1+exp((2-a)*t/tau));
end
subplot(1,size(a_set,2),find(a==a_set))
hold on
plot(x(a==a_set,:),y(a==a_set,:))
xlim([-5 5]) 
ylim([-5 5]) 
end

%% 2 b
a = 1;
A = [-a,-2;-(3-a),-1];
h_0 = [1;-1];
x_vec = -5:.1:5;
y_vec = -5:.1:5;
E = -.5.*[x_vec;y_vec]'*A*[x_vec;y_vec];

figure('position',[50 50 350 350])
imagesc(E)
set(gca,'xtick',1:10:size(x_vec,2),'xticklabel',x_vec(1:10:end),...
    'ytick',1:10:size(y_vec,2),'yticklabel',y_vec(1:10:end))
xlabel('distance relative to fixed point in x')
ylabel('distance relative to fixed point in y')
axis square
c = colorbar;
title(c,'energy')