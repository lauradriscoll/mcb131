
gaus = @(X,s,m) exp(-0.5 * ((X - m)./s).^2) ./ (sqrt(2*pi) .* s);
lle = @(X,N,s,m) (-N/2)*log(2*pi) - N*log(s) - 1./(2*s'*s)*(X - m)'*(X - m);

X_1D = -5:.1:5;
X_set = cat(1,X_1D,X_1D);
N = size(X_set,1);
g = nan(size(X_set,1));

sigma1 = cov(rand(100,2))+rand;
mu1 = 5*rand(1,N)-2.5;

sigma2 = cov(rand(100,2))+rand;
mu2 = 5*rand(1,N)-2.5;

thresh = rand;

figure;
subplot(1,2,1)
hold on
axis square
[X1,X2] = meshgrid(X_1D,X_1D);
F = mvnpdf([X1(:) X2(:)],mu1,sigma1) - mvnpdf([X1(:) X2(:)],mu2,sigma2) -thresh;
F = reshape(F,length(X_1D),length(X_1D));

mesh(X_1D,X_1D,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([min(X_1D) max(X_1D) min(X_1D) max(X_1D) min(F(:)) max(F(:))])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
view(30, 30)

C = mvnpdf([X1(:) X2(:)],mu1,sigma1) > mvnpdf([X1(:) X2(:)],mu2,sigma2);
C = reshape(C,length(X_1D),length(X_1D))+0;

x1 = repmat(X_1D,size(X_1D,2),1);
x2 = repmat(X_1D',1,size(X_1D,2));
subplot(1,2,2)
hold on
axis square
imagesc('XData',X_1D,'YData',X_1D,'CData',C)
scatter(x1(:),x2(:),[],F(:)-min(F(:)))
axis([min(X_1D) max(X_1D) min(X_1D) max(X_1D)])
xlabel('x1'); ylabel('x2');

% s1 = [3 1;1 3];
% s2 = [1 .5;.5 1];
% m1 = [5 10];
% m2 = [-5 10];
% N = 2;
% cmap = [1 0 0;0 0 1;0 0 0];
% 
% figure;
% hold on
% plot(m1(1),m1(2),'o','color',cmap(1,:))
% plot(m2(1),m2(2),'o','color',cmap(2,:))
% 
% for samples = 1:100
% X = randi([-10 10],1,N);
% y= -N*log(s1)+N*log(s2) - (2*s1^2)^-1*sum(X-m1).^2 + (2*s2^2)^-1*sum(X-m2).^2;
% plot(X(1),X(2),'o','color',cmap(1+(y==1),:))
% end

%% 3.2
n_reps = 50;
N_set = [10 20 100];
P_set = 10;
e_con = nan(size(P_set,2),size(N_set,2),n_reps);

for N = N_set;
for P = P_set;
X = rand(N,P);
y0 = randi([-1 1],P,1);

w_all = [];
e_all = [];
for rep = 1:n_reps
[w, converged, epochs, error_history] = perceptron(X,y0);
w_all = cat(2,w_all,w);
e_all = cat(2,e_all,error_history);

e_con(P==P_set,N==N_set,rep) = epochs;
end
end
end

figure('position',[50 50 300 300]);
hold on
plot(N_set,squeeze(nanmean(e_con(1,:,:),3)),'or')
plot(N_set,squeeze(e_con(1,:,:)),'.k')
legend('mean','all data','location','southeast')
ylabel('Number of Epochs')
xlabel('N')
xlim([0 110])
large_text
axis square

%% 3.3
n_reps = 20;
N_set = [5 20 100];
P_set = .5:.5:2.5;
e_con = nan(size(P_set,2),size(N_set,2),n_reps);
converged_all = nan(size(P_set,2),size(N_set,2),n_reps);
legend_labs = [];

figure;
hold on
cmap = parula(1+size(N_set,2));

for N = N_set;
P_set_current = round(N*P_set);
for P = P_set_current;
for rep = 1:n_reps
X = rand(N,P);
y0 = randi([-1 1],P,1);
[~, converged, epochs, ~] = perceptron(X,y0);
converged_all(P==P_set_current,N==N_set,rep) = converged;
e_con(P==P_set_current,N==N_set,rep) = epochs;
end
plot((P/N),squeeze(nanmean(converged_all(P==P_set_current,N==N_set,:))),'o','color',cmap(N==N_set,:))
drawnow
end
legend_labs = cat(2,legend_labs,{['N = ' num2str(N)]});
end
ylabel('fraction converged')
xlabel('alpha')
xlim([0 3])
large_text
legend(num2str(N_set(1)))
axis square


%% 3.4a-b ADAPTRON
% Adatron
% X [ NxP] : P samples of dim=N
% y0 [Px1]: labels vector of P samples
% alpha [Px1] : SVM coef

load('/Users/Laura/Desktop/MCB131/code/data_10D.mat')

n_reps = 50;
P_set = [10 50 100 300];
perc_train = nan(size(P_set,2),n_reps);
perc_test = nan(size(P_set,2),n_reps);
num_svs = nan(size(P_set,2),n_reps);

for P = P_set;
    for rep = 1:n_reps
        ind_all = 1:size(X,2);
        ind_train = ind_all(randperm(size(X,2),P));
        ind_test = ind_all(~ismember(1:size(X,2),ind_train));
        
        X_train = X(:,ind_train);
        y_train = y0(ind_train);
        
        X_test = X(:,ind_test);
        y_test = y0(ind_test);
        
        alpha = adatron( X_train , y_train);
        num_svs(P==P_set,rep) = sum(alpha>0);
        
        w =  (alpha' .* y_train' ) * X_train';
        
        pred_train = sign(w*X_train)';
        pred_test = sign(w*X_test)';
        
        perc_train(P==P_set,rep) = sum(pred_train - y_train == 0)/size(y_train,1);
        perc_test(P==P_set,rep) = sum(pred_test - y_test == 0)/size(y_test,1);
    end
end

figure;
hold on
plot(P_set,num_svs,'.k')
plot(P_set,nanmean(num_svs,2),'o-r')
axis square
title('adaptron')
ylabel('number of support vectors')
xlabel('P')

figure;
subplot(1,2,1)
hold on
plot(P_set,perc_test,'.k')
plot(P_set,nanmean(perc_test,2),'o-r')
axis square
title('adaptron')
ylabel('testing accuracy')
xlabel('P')

%% 3.4c

load('/Users/Laura/Desktop/MCB131/code/data_10D.mat')

n_reps = 50;
P_set = [10 50 100 300];
perc_train = nan(size(P_set,2),n_reps);
perc_test = nan(size(P_set,2),n_reps);
num_svs = nan(size(P_set,2),n_reps);

for P = P_set;
    for rep = 1:n_reps
        ind_all = 1:size(X,2);
        ind_train = ind_all(randperm(size(X,2),P));
        ind_test = ind_all(~ismember(1:size(X,2),ind_train));
        
        X_train = X(:,ind_train);
        y_train = y0(ind_train);
        
        X_test = X(:,ind_test);
        y_test = y0(ind_test);
        
        [w, ~, ~, ~] = perceptron(X_train,y_train);
        
        pred_train = sign(w'*X_train)';
        pred_test = sign(w'*X_test)';
        
        perc_train(P==P_set,rep) = sum(pred_train - y_train == 0)/size(y_train,1);
        perc_test(P==P_set,rep) = sum(pred_test - y_test == 0)/size(y_test,1);
    end
end

subplot(1,2,2)
hold on
plot(P_set,perc_test,'.k')
plot(P_set,nanmean(perc_test,2),'o-r')
axis square
title('perceptron')
ylabel('testing accuracy')
xlabel('P')
