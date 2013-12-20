function plot_sepshock_decision(Y, U, P,P2, tname)
% plot the loss, probabilities, true label, predicted label
%
% Y Tx1 true labels
% U Tx1 decisions made
% P TxC
% title title of graph


figure;

subplot(4,1,1);
title(tname);
ylabel('SVM Decision Value');
hold on;
plot(P(:,1), ':', 'Color', [0 128 255]/255, 'LineWidth', 2);
plot(P(:,2), ':', 'Color', [127 0 255]/255, 'LineWidth', 2);


subplot(4,1,2);
hold on;
ylabel('CRF Probability');
plot(P2(:,1), ':', 'Color', [0 128 255]/255, 'LineWidth', 2);
plot(P2(:,2), ':', 'Color', [127 0 255]/255, 'LineWidth', 2);
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

nosep = find(Y == -1);
unsure = find(Y == 0);
yessep = find(Y == 1);

u1ind = setdiff(find(U == 1),nosep);
u3ind = setdiff(find(U == 2),yessep);
u5ind = find(U == 3);
u1ind2 = intersect(find(U == 1),nosep);
u3ind2 = intersect(find(U == 2),yessep);

subplot(4,1,3);
hold on;
ylabel('True Label')
plot(nosep,ones(size(nosep))*-.05,'+', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(unsure,ones(size(unsure))*-.05,'+', 'Color', [255 0 125]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(yessep,ones(size(yessep))*-.05,'+', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);


subplot(4,1,4);
hold on;
plot(u5ind,ones(size(u5ind))*-.35,'+', 'Color', [0 204 0]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u1ind,ones(size(u1ind))*-.15,'+', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u3ind,ones(size(u3ind))*-.25,'+', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u3ind2,ones(size(u3ind2))*-.25,'+', 'Color', [127 0 255]/255, 'MarkerSize',10, 'LineWidth', 2);
plot(u1ind2,ones(size(u1ind2))*-.15,'+', 'Color', [0 128 255]/255, 'MarkerSize',10, 'LineWidth', 2);

xlabel('Observation');
ylabel('Model Decisions');
end