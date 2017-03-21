
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