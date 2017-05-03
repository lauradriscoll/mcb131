function saveFormattedFig(figName)
% Save a word-document friendly figure.
%
% Figure making in matlab is awful, we recommend avoiding modifing this
% unless you want to change something simple or strongly dislike the
% figure.

% Change figure properties to hopefully be cross platform pretty
% set(gcf,...
%     'Position',[5 5 480*2 480*2 ],...   % Dimensions in matlab
%     'PaperUnits','inches',...           % Inches in America
%     'PaperSize',[4*2 4*2],...           % A size that fits in word doc
%     'Color', [1 1 1]);                  % White background

% Change all font sizes
set(findall(gcf,'-property','FontSize'),'FontSize',14)

% Remove silly boxes
set(findall(gcf,'-property','Box'),'Box','off')

% Save as png file using print
print(gcf,figName,'-dpng')