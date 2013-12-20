function plot_reclas_decision(X, Y, U, P, V, tname)
% plot the loss, probabilities, true label, predicted label
%
% X Tx1 observations
% Y Tx1 true labels
% U Tx1 decisions made
% P TxC
% S C+1xC cost matrix
% title title of graph

C = size(P,2);

figure;
%plot(X/(max(X)),'k.-');
hold on;
plot(P(:,1), ':', 'Color', [0 128 255]/255, 'LineWidth', 2);
plot(P(:,2), ':', 'Color', [0 0 255]/255, 'LineWidth', 2);
plot(P(:,3), ':', 'Color', [127 0 255]/255, 'LineWidth', 2);
plot(P(:,4), ':', 'Color', [255 0 127]/255, 'LineWidth', 2);

% % Cost of truth
% CT = zeros(size(U));
% CT(1) = V(1,Y(1));
% % Cost of decision
% CD = zeros(size(U));
% CD(1) = min(V(1,:));
% 
% for t=2:length(U)
%     CT(t) = CT(t-1)+ V(t,Y(t));
%     CD(t) = CD(t-1) + min(V(t,:));
% end
% maxC = max(CT(t),CD(t));
% plot(CD/maxC,'--', 'Color', [204 0 0]/255, 'LineWidth', 1);
% plot(CT/maxC,'--', 'Color', [102 0 102]/255, 'LineWidth', 1);

c1ind = find(Y == 1);
c2ind = find(Y == 2);
c3ind = find(Y == 3);
c4ind = find(Y == 4);

plot(c1ind,ones(size(c1ind))*-.05,'+', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(c2ind,ones(size(c2ind))*-.05,'+', 'Color', [0 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(c3ind,ones(size(c3ind))*-.05,'+', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(c4ind,ones(size(c4ind))*-.05,'+', 'Color', [255 0 127]/255, 'MarkerSize',10, 'LineWidth', 2);

u1ind = setdiff(find(U == 1),c1ind);
u2ind = setdiff(find(U == 2),c2ind);
u3ind = setdiff(find(U == 3),c3ind);
u4ind = setdiff(find(U == 4),c4ind);
u5ind = find(U == 5);
u1indc = intersect(find(U == 1),c1ind);
u2indc = intersect(find(U == 2),c2ind);
u3indc = intersect(find(U == 3),c3ind);
u4indc = intersect(find(U == 4),c4ind);

plot(u1ind,ones(size(u1ind))*-.15,'o', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u2ind,ones(size(u2ind))*-.25,'o', 'Color', [0 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u3ind,ones(size(u3ind))*-.35,'o', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u4ind,ones(size(u4ind))*-.45,'o', 'Color', [255 0 127]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u5ind,ones(size(u5ind))*-.55,'o', 'Color', [0 204 0]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u1indc,ones(size(u1indc))*-.15,'+', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u2indc,ones(size(u2indc))*-.25,'+', 'Color', [0 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u3indc,ones(size(u3indc))*-.35,'+', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u4indc,ones(size(u4indc))*-.45,'+', 'Color', [255 0 127]/255, 'MarkerSize',10, 'LineWidth', 2);
title(tname);

xlabel('Observation');
% lnames = cell(16);
% lnames{1} = 'Obsv';
% lnames{2} = 'Prob C1';
% lnames{3} = 'Prob C2';
% lnames{4} = 'Prob C3';
% lnames{5} = 'Prob C4';
% lnames{6} = 'Dec Cost';
% lnames{7} = 'True Cost';
% lnames{8} = 'True C1';
% lnames{9} = 'True C2';
% lnames{10} = 'True C3';
% lnames{11} = 'True C4';
% lnames{12} = 'Dec C1';
% lnames{13} = 'Dec C2';
% lnames{14} = 'Dec C3';
% lnames{15} = 'Dec C4';
% lnames{16} = 'IDK';
%legend(lnames{1},lnames{2}, lnames{3});
end