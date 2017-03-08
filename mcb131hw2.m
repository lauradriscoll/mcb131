% %% 1C
% 
% load('/Users/Laura/Downloads/Rho_vs_ep.mat')
% 
% %PART I
% L = 101;
% kmin = 2*pi/L;
% cmap = jet(size(ep_vec,2));
% 
% k_set = 1/L:1/L:2*pi;
% w_set = nan(size(ep_vec,2),size(k_set,2));
% 
% for k = k_set
% Ck = (k^2+kmin^2).^(-.9);
% c = Ck./ep_vec;
% xk = 1./(2*(c+1)) .* (-(c+2) + sqrt((c+2).^2 + 4*(c./Rho_vec - c - 1)));
% w_set(:,k==k_set) = sqrt(xk);
% end
% 
% figure
% hold on
% l_set = [];
% for ind = 1:size(ep_vec,2)
% plot(k_set,w_set(ind,:),'-','color',cmap(ind,:),'lineWidth',2)
% l_set = cat(1,l_set,{['eps = ' num2str(ep_vec(ind))]});
% end
% legend(l_set)
% xlabel('k_x')
% ylabel('|w_k|')
% xlim([0 max(k_set)])
% 
% %PART II
% figure('position', [50 50 1100 300]);
% hold on
% k_set_short = k_set;
% ind_set = [1 10];
% 
% for ind = ind_set
%     
%     Ck = (k_set_short.^2+kmin^2).^(-.9);
%     c = Ck./ep_vec(ind);
%     xk = 1./(2*(c+1)) .* (-(c+2) + sqrt((c+2).^2 + 4*(c./Rho_vec(ind) - c - 1)));
%     w_set = sqrt(xk);
%     
%     for x = 0:.01:1;
%         if x==0
%             max_resp = w_set*cos(x*k_set)';
%         end
%         subplot(1,2,find(ind == ind_set))
%         hold on
%         plot(x,(w_set*cos(x*k_set)')/max_resp,'.','color',cmap(ind,:),'MarkerSize',10)
%     end
%     
%     if ind==min(ind_set)
%         title(['low noise : eps = ' num2str(ep_vec(ind))])
%     else
%         title(['high noise : eps = ' num2str(ep_vec(ind))])
%     end
%     
%     ylim([-1 1])
%     xlabel('x in real space')
%     ylabel('w(r)')
% end
% 
% %PART III
% figure;
% hold on
%     
% %     k = k_set(randperm(size(k_set,2),1)); %choose any k
% 
% lagrange_constraint = nan(size(ep_vec,2),size(k_set_short,2));
% 
% for k = k_set_short
%     Ck = (k^2+kmin^2).^(-.9);
%     c = Ck./ep_vec;
%     xk = 1./(2*(c+1)) .* (-(c+2) + sqrt((c+2).^2 + 4*(c./Rho_vec - c - 1)));
%     lagrange_constraint(:,k==k_set_short) = xk.*(c+1)+1-4;
% end
% 
% for ind = 1%1:size(ep_vec,2) %plot for various eps
%     plot(k_set_short,lagrange_constraint(ind,:),'-','color',cmap(ind,:),'lineWidth',2)
% end
% 
% xlabel('k_x')
% ylabel('lagrange constraint')
% xlim([0 max(k_set)])
% 
% nanmean(lagrange_constraint(ind,:))


%% Problem 2B
N = 100;
a_set = [-.45:.2:.45];
cmap = parula(size(a_set,2));

figure;

subplot(1,2,1)
hold on
l_set = [];
for a = a_set;

A_vec = [1 a zeros(1, N-3) a];
A = [];
j_set = 1:N;
for n = 1:N
    A = cat(1,A,circshift(A_vec,n-1,2));
end

A = A(1:N,1:N);
plot(j_set,1+2*a*cos((2/N)*pi.*j_set),'o','color',cmap(a==a_set,:))

l_set = cat(1,l_set,{['a = ' num2str(a)]});
end
xlabel('j')
ylabel('analytical eigenvalues')
axis square

%verify w eig
subplot(1,2,2)
hold on
l_set = [];
for a = a_set;

A_vec = [1 a zeros(1, N-3) a];
A = [];
for n = 1:N
    A = cat(1,A,circshift(A_vec,n-1,2));
end

A = A(1:N,1:N);
plot(eig(A),'o','color',cmap(a==a_set,:))

l_set = cat(1,l_set,{['a = ' num2str(a)]});
end
legend(l_set,'location','southeast')
xlabel('sorted ascending eigenvalues')
ylabel('empirical eigenvalues')
axis square

% 
% %% Problem 3
% 
% n_draw = 10000;
% r = nan(2,n_draw);
% 
% for ind = 1:n_draw
%     
% x_set = rand(100,1)*4+1;
% y_set = ones(size(x_set));
% 
% rand_n = rand.*randi(2,1);
% x_match = x_set(randi(size(x_set,2)))+rand_n;
% y_match = (rand_n+1)*y_set(randi(size(x_set,2)));
% r(:,ind) = [x_match;y_match];
% end
% figure;plot(r(1,:),r(2,:),'ok')
% 
% mu = [mean(r(1,:)) mean(r(2,:))];
% sig2 = cov(r');
% 
% 

% 
% 
% %%Problem 4
% load('/Users/Laura/Downloads/mixed_images.mat')
% cmap = hsv(3);
% nr = 2;
% nc = 3;
% 
% Z = [mixed1(:)'; mixed2(:)'; mixed3(:)'];
% size_im = size(mixed1);
% 
% figure;
% subplot (nr,4,1)
% hold on
% plot(1:3, mean(Z,2),'ok')
% ylabel('Mean')
% xlim([0 4])
% set(gca,'xtick',1:3,'xticklabel',{'Image 1','Image 2','Image 3'},'XTickLabelRotation',45)
% 
% subplot (nr,4,2)
% hold on
% plot(1:3, var(Z,0,2),'ok')
% ylabel('Variance')
% xlim([0 4])
% set(gca,'xtick',1:3,'xticklabel',{'Image 1','Image 2','Image 3'},'XTickLabelRotation',45)
% 
% subplot (nr,4,3)
% hold on
% plot(1:3, skewness(Z,0,2),'ok')
% ylabel('Skewness')
% xlim([0 4])
% set(gca,'xtick',1:3,'xticklabel',{'Image 1','Image 2','Image 3'},'XTickLabelRotation',45)
% 
% subplot (nr,4,4)
% hold on
% plot(1:3, kurtosis(Z,0,2),'ok')
% ylabel('Kurtosis')
% xlim([0 4])
% set(gca,'xtick',1:3,'xticklabel',{'Image 1','Image 2','Image 3'},'XTickLabelRotation',45)
% 
% for x = 1:3
% subplot (nr,3,3+x)
% imagesc(reshape(Z(x,:),size_im))
% set(gca,'xtick',[],'ytick',[])
% title(['Image ' num2str(x)])
% colormap(gray)
% end
% 
% 
% % Center data
% mu = mean(Z,2);
% Z_center = bsxfun(@minus,Z,mu);
% 
% % Whiten data
% R = cov(Z_center');
% [U, S, ~] = svd(R,'econ');
% T  = U * diag(1 ./ sqrt(diag(S))) * U';
% Z_sphere = T * Z_center;
% 
% it = 0;
% maxit = 10;
% n_sources = 3;
% 
% normVec = @(X) bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));
% W = normVec(rand(n_sources,size(Z,1))); % Random initial weights
% it_kurt = nan(n_sources,maxit);
% 
% while it < maxit
%     it = it + 1;
%     
%     Z_sep = W * Z_sphere;
%     it_kurt(:,it) = kurtosis(Z_sep,1,2);
%     
%     W_old = W; % Save last weights
%     
%     wz = permute(W(1,:) * Z_sphere,[1, 3, 2]);
%         kurt_num = wz.^3;
%         kurt_denom = wz.^2;
%         
%     W(1,:) = mean(bsxfun(@times,kurt_num,permute(Z_sphere,[3, 1, 2])),3) - ...
%              bsxfun(@times,mean(kurt_denom,3),W(1,:));
%          
%     wz = permute(W(2,:) * Z_sphere,[1, 3, 2]);
%         kurt_num = wz.^3;
%         kurt_denom = wz.^2;
%          
%     W(2,:) = mean(bsxfun(@times,kurt_num,permute(Z_sphere,[3, 1, 2])),3) - ...
%              bsxfun(@times,mean(kurt_denom,3),W(2,:));
%          
%     wz = permute(W(3,:) * Z_sphere,[1, 3, 2]);
%         kurt_num = wz.^3;
%         kurt_denom = wz.^2;
%          
%     W(3,:) = mean(bsxfun(@times,kurt_num,permute(Z_sphere,[3, 1, 2])),3) - ...
%              bsxfun(@times,mean(kurt_denom,3),W(3,:));
%          
%     W_norm = normVec(W);
%     
%     % Decorrelate weights
%     [U, S, ~] = svd(W,'econ');
%     W = U * diag(1 ./ diag(S)) * U' * W;
% end
% 
% figure
% nr = 3;
% for x = 1:3
% subplot(nr,2,2*x)
% imagesc(reshape(Z_sep(x,:),size_im));
% set(gca,'xtick',[],'ytick',[])
% colormap(gray)
% subplot(nr,2,2*x-1)
% plot(it_kurt(x,:),'ok')
% xlabel('iteration number')
% ylabel('kurtosis')
% end






