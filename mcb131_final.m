
%% Problem 4: Reinforcement learning on a ring
%% 4b
N = 100;
p = .2;
g_set = [.1 .9 .99];

s_r = 50;%randi(N);
r = -.1*ones(N,3); % 1 s-1 % 2 s % 3 s+1

r(mod(s_r,N),2) = r(mod(s_r,N),2)+1;
r(mod(s_r-1,N),3) = r(mod(s_r-1,N),3)+1;
r(mod(s_r+1,N),1) = r(mod(s_r+1,N),1)+1;

its = 600;
V = rand(size(g_set,2),N,its);
s_set = 0:N-1;

for g = g_set
    for it = 1:its-1
    for s = s_set(randperm(N))
        
        s1 = mod(s-1,N)+1;
        s2 = mod(s,N)+1;
        s3 = mod(s+1,N)+1;
        
        discounted_s1 = g*(p*V(g == g_set,mod(s1-2,N)+1,it)+ (1 - 2*p)*V(g == g_set,mod(s1-1,N)+1,it)+...
            p*V(g == g_set,mod(s1,N)+1,it));
        discounted_s2 = g*(p*V(g == g_set,mod(s2-2,N)+1,it)+(1 - 2*p)*V(g == g_set,mod(s2-1,N)+1,it)+...
            p*V(g == g_set,mod(s2,N)+1,it));
        discounted_s3 = g*(p*V(g == g_set,mod(s3-2,N)+1,it)+(1 - 2*p)*V(g == g_set,mod(s3-1,N)+1,it)+...
            p*V(g == g_set,mod(s3,N)+1,it));
        
        V(g == g_set,s2,it+1) = p*(r(s2,1)+discounted_s1) + p*(r(s2,3)+discounted_s3) +...
            (1 - 2*p)*(r(s2,2)+discounted_s2);
    end
    end
end

%% visualize 4b
figure('position',[0 0 1200 1000]);
for g_ind = 1:3
subplot(3,2,g_ind*2-1)
plot(V(g_ind,:,end)','o-','lineWidth',1.1)
set(gca,'xtick',(s_r-50):25:(s_r+50),'xticklabel',...
    {'s_r - 50','s_r - 25','s_r','s_r + 25','s_r + 50'})
xlabel('state')
ylabel('value')
title(['gamma = ' num2str(g_set(g_ind))])
subplot(3,2,g_ind*2)
imagesc(squeeze(V(g_ind,:,:)))
xlabel('iteration')
ylabel('state')
title(['gamma = ' num2str(g_set(g_ind))])
set(gca,'ytick',(s_r-50):25:(s_r+50),'yticklabel',...
    {'s_r - 50','s_r - 25','s_r','s_r + 25','s_r + 50'})
h = colorbar;
title(h,'value')
end
saveFormattedFig(fullfile('C:\Users\Laura\Desktop\mcb 131\code\mcb131\final_4b'))

%% 4c
N = 100;
g_set = [.1 .9 .99];

s_r = 50;%randi(N);
r = -.1*ones(N,3); % 1 s-1 % 2 s % 3 s+1

r(mod(s_r,N),2) = 0;
r(mod(s_r-1,N),3) = 0;
r(mod(s_r+1,N),1) = 0;

its = 1000;
V = rand(size(g_set,2),N,its);
G = rand(N,1);
s_set = 0:N-1;

for g = g_set
    for it = 1:its-1
    for s = s_set(randperm(N))
        
        s1 = mod(s-1,N)+1;
        s2 = mod(s,N)+1;
        s3 = mod(s+1,N)+1;
        
        discounted_s1 = g*(p*V(g == g_set,mod(s1-2,N)+1,it)+ (1 - 2*p)*V(g == g_set,mod(s1-1,N)+1,it)+...
            p*V(g == g_set,mod(s1,N)+1,it));
        discounted_s2 = g*(p*V(g == g_set,mod(s2-2,N)+1,it)+(1 - 2*p)*V(g == g_set,mod(s2-1,N)+1,it)+...
            p*V(g == g_set,mod(s2,N)+1,it));
        discounted_s3 = g*(p*V(g == g_set,mod(s3-2,N)+1,it)+(1 - 2*p)*V(g == g_set,mod(s3-1,N)+1,it)+...
            p*V(g == g_set,mod(s3,N)+1,it));
        
        V(g == g_set,s2,it+1) = p*(r(s2,1)+discounted_s1) + p*(r(s2,3)+discounted_s3) +...
            (1 - 2*p)*(r(s2,2)+discounted_s2);
    end
    end
end


%%

%% Problem 5: Ring Attractor
%% initialize
N = 100;
h0 = 2;
h1 = 1;
T = 2;
tau = 1;
nsteps = 1000;
step_size = .01;
theta = 2*pi.*(1:N)/N;
w0_set = [0 1 2];
w1_set = [-10 2 10];
fig_plot = figure;
fig_im = figure;
nr = size(w0_set,2);
nc = size(w1_set,2);
plot_Wij = 0;
add_noise = 1;
theta0 = pi;

%% simuluate dynamics for various w0 w1
for w0 = w0_set
for w1 = w1_set
%% make weight matrix
Wij = nan(N);

if add_noise==1
a = rand(N,nsteps);
else
a = zeros(N,nsteps);
end

for i  = 1:100
for j  = 1:100
    if i==j
Wij(i,i) = 0;
    else
Wij(i,j) = (1/N)*(w0+w1*cos(theta(i) - theta(j)));
    end
end
end

%% plot weight matrix
if plot_Wij==1
imagesc(Wij)
axis square
title('Wij')
h = colorbar;
title(h,'synaptic weight')
xlabel('neuron i')
ylabel('neuron j')
end

%% iterate through time
step_set = step_size:step_size:nsteps*step_size;
for t = step_set
    for i = 1:N
        f = Wij(i,:)*a(:,t==step_set)+h0-T+h1*cos(theta(i)-theta0);
        if f<0
            f = 0;
        end
        a(i,find(t==step_set)+1) = a(i,t==step_set) - ...
            a(i,t==step_set)*step_size + f*step_size;
    end
end

%% visualizations
cmap = hsv(100);
figure(fig_plot)
subplot(nr,nc,(find(w0_set==w0)-1)*nc+find(w1_set==w1))
    hold on
if w0==w0_set(1) && w1==w1_set(end)
for i = 10:10:N
    plot((a(i,:)),'color',cmap(i,:),'lineWidth',1.5)
end
legend('neuron 10','neuron 20','neuron 30','neuron 40','neuron 50',...
    'neuron 60','neuron 70','neuron 80','neuron 90','neuron 100')
end
for i = 1:N
    plot((a(i,:)),'color',cmap(i,:),'lineWidth',1.5)
end
xlim([0 nsteps])
xlabel('time steps')
ylabel('activity')
title(['w0 = ' num2str(w0) ' ; w1 = ' num2str(w1)])

figure(fig_im)
subplot(nr,nc,(find(w0_set==w0)-1)*nc+find(w1_set==w1))
    imagesc(a)
axis square
title(['w0 = ' num2str(w0) ' ; w1 = ' num2str(w1)])
h = colorbar;
title(h,'activity')
ylabel('neuron i')
xlabel('time steps')
end
end
figure(fig_plot)
saveFormattedFig(fullfile('C:\Users\Laura\Desktop\mcb 131\code\mcb131\final_5a'))
figure(fig_im)
saveFormattedFig(fullfile('C:\Users\Laura\Desktop\mcb 131\code\mcb131\final_5b'))
