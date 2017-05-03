% % %% simulations troubleshooting 1
% % a_set = .3;%.01:.01:1;
% % n = 10000;
% % v = nan(n,1);
% % for a = a_set
% % S_mu = sign(binornd(1,a,n,1)-.5);
% % S_nu = sign(binornd(1,a,n,1)-.5);
% %
% % %v(a==a_set) = nanmean((S_mu'*S_mu*S_nu)/n);
% % v(a==a_set) = S_mu.*nanmean(S_mu)/n;
% % end
% % figure;
% % plot(a_set,v)
% % hold on
% % plot(a_set,a_set.^2)
% % plot(a_set,a_set.^3)
% %
% % %%
% %
% %
% close all
% %% 2 a
% tau = .1;
% v0 = .1;
% a_set = [-2 2.001 4];
% tmax = 100;
% x = nan(size(a_set,2),tmax);
% y = nan(size(a_set,2),tmax);
% 
% for a=a_set
% b = tau*v0/(a-2);
% for t = 1:tmax
% x(a==a_set,t) = -b*(-1+exp((2-a)*t/tau));
% y(a==a_set,t) = -b*(-1+exp((2-a)*t/tau));
% end
% subplot(1,size(a_set,2),find(a==a_set))
% hold on
% plot(x(a==a_set,:),y(a==a_set,:))
% xlim([-5 5]) 
% ylim([-5 5]) 
% end
% saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb2a'))
% 
% %% 2 b
% a = 1;
% A = [-a,-2;-(3-a),-1];
% tau_set = [1 .3];
% v0_set = [1 -3];
% x_vec = -5:.1:5;
% y_vec = -5:.1:5;
% figure('position',[50 50 800 350])
% 
% for ind = 1
%     tau = tau_set(ind);
%     v0 = v0_set(ind);
% E = [x_vec;y_vec]'*A*[x_vec;y_vec] - tau*v0*(repmat(x_vec,size(x_vec,2),1)'-...
%     repmat(y_vec',1,size(y_vec,2))');
% 
% % subplot(1,2,ind)
% imagesc(E)
% set(gca,'xtick',1:10:size(x_vec,2),'xticklabel',x_vec(1:10:end),...
%     'ytick',1:10:size(y_vec,2),'yticklabel',y_vec(1:10:end))
% xlabel('distance relative to fixed point in x')
% ylabel('distance relative to fixed point in y')
% % title(['tau = ' num2str(tau) ' ; ' 'v0 = ' num2str(v0)])
% axis square
% c = colorbar;
% title(c,'energy')
% end
% saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb2b'))

%% 3
load('/Users/Laura/Desktop/MCB131/code/values_opt.mat')

w=12; h=4;
alpha = 1;
nsteps = 100000;
e_set = 0:.1:1;
er_all = nan(size(e_set,2),nsteps);
c_time = nan(size(e_set,2),2);
policy_diff = nan(size(e_set,2),nsteps);
sum_goal = nan(size(e_set,2),1);
mean_g = nan(size(e_set,2),1000);

for e = e_set;
p = nan(2,nsteps);
re = nan(1,nsteps);
er = nan(1,nsteps);
er_opt = nan(1,nsteps);
g = zeros(1,nsteps); 
p(:,1) = [1 4];
policy = randi(4,w,h);
values=zeros(w,h,4); % initial value = set at zeros.
reward=-ones(w,h); % reward values.
reward(2:end-1,4) = -100;
reward(12,4) = 0;
transition = [-1 0; 1 0; 0 1; 0 -1];


for step = 1:nsteps
    %% choose policy
    if rand>e %random policy
        a = randi(4);
        
    else %optimal policy
        a = policy(p(1,step),p(2,step));
    end
    
    %% update step
    p(:,step+1) = p(:,step) + transition(a,:)';
    
    % stay within bounds
    if p(1,step+1)>w
        p(1,step+1) = w;
    elseif p(1,step+1)<1
        p(1,step+1) = 1;
    end
    
    if p(2,step+1)>h
        p(2,step+1) = h;
    elseif p(2,step+1)<1
        p(2,step+1) = 1;
    end
    
    %% update value
    re(step) = reward(p(1,step+1),p(2,step+1));
    er(step) = reward(p(1,step+1),p(2,step+1))+ max(values(p(1,step+1),p(2,step+1),:)) - ...
        values(p(1,step),p(2,step),a);
    er_opt(step) = reward(p(1,step+1),p(2,step+1))+ max(values(p(1,step+1),p(2,step+1),:)) - ...
        values_opt(p(1,step),p(2,step),a);
    
    values(p(1,step),p(2,step),a) = values(p(1,step),p(2,step),a) + alpha*er(step);
    
    temp_a = find(values(p(1,step),p(2,step),:) == max(values(p(1,step),p(2,step),:)));
    policy(p(1,step),p(2,step)) = temp_a(randi(size(temp_a,1)));%temp_a(1);%
    
    %% send back to start if falls off cliff
    if p(1,step+1)>1 && p(2,step+1)==h
        if p(1,step+1)==w && p(2,step+1)==h
        g(1,step) = max(g(1,:))+1;
        
        if g(1,step)<1000
        mean_g(e == e_set,g(1,step)) = nanmean(re(find(g==(g(1,step)-1)):step));
        end
        
        end
        p(:,step+1) = p(:,1);
    end
    pd_temp = abs(values - values_opt);
    policy_diff(e==e_set,step) = sum(pd_temp(:));
end

%% save policy at limits
    if e==0
    lim_policy{1} = policy;
    elseif e==1
    lim_policy{2} = policy;
    end
    
%% save convergence and goal statistics
sum_goal(e==e_set,1) = sum(g>0);
er_all(e==e_set,:) = movmean(policy_diff(e==e_set,:),1000);
c_time(e==e_set,1) = min([(find(movmean(policy_diff(e==e_set,:),1000)>101,1,'last')) nsteps]);
c_time(e==e_set,2) = max(g(1,1:c_time(e==e_set,1)));
end

%% plotting
figure('position',[50 500 1000 400]);
subplot(1,2,1)
hold on
plot(c_time(:,1),'-o')
ylabel('number of steps to convergence')
xlabel('epsilon')
set(gca,'xtick',1:2:round(size(e_set,2)),'xticklabel',e_set(1:2:end))

subplot(1,2,2)
hold on
plot(c_time(:,2),'-o')
set(gca,'xtick',1:2:round(size(e_set,2)),'xticklabel',e_set(1:2:end))
ylabel('number of goals to convergence')
xlabel('epsilon')       
saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb3c1'))

figure('position',[50 50 500 400]);
imagesc(er_all(:,1:10000),[-1 0])
xlabel('steps')
ylabel('epsilon')
set(gca,'ytick',1:2:round(size(e_set,2)),'yticklabel',e_set(1:2:end))
c = colorbar;
ylabel(c,'error moving average (window = 1000 steps)')
saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb3c2'))
%%
figure('position',[50 500 1000 400]);
subplot(1,2,1)
imagesc(mean_g)
xlabel('goal number')
ylabel('epsilon')
set(gca,'ytick',1:2:round(size(e_set,2)),'yticklabel',e_set(1:2:end))
xlim([0 1000])
c = colorbar;
title(c,'mean rewards / episode')

subplot(1,2,2)
plot(mean_g(1:5:end,1:50)','-o')
xlabel('goal number')
ylabel('mean rewards / episode')
legend({num2str(e_set(1));num2str(e_set(6));num2str(e_set(11))},'location','southEast')
saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb3c3'))

figure;plot(policy_diff(1:5:end,1:10000)','-o')
ylabel('sum difference from optimal policy')
xlabel('steps')
legend({num2str(e_set(1));num2str(e_set(6));num2str(e_set(11))},'location','southEast')
saveFormattedFig(fullfile('\Users\Laura\Desktop\MCB131\hw5pb3c4'))