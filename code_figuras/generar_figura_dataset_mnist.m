clear; close all; clc; 
test = load('mnist_test.csv');
images = test(1:64,2:785); clear test;
images = images'/255;
images = reshape(images, [28, 28, 64]);
images = permute(images, [2 1 3]);
s = [64, 64];
matriz = zeros(s(1)*4,s(2)*4);
indices = randi(64, 16);
for indx = 0:15
    columna = mod(indx,4);
    fila = floor((indx)/4);
    matriz(fila*s(1)+1:s(1)*(fila+1),columna*s(2)+1:s(2)*(columna+1)) = imresize(images(:,:,indices(indx+1)),s);
end

figure('units','normalized','outerposition',[0 0 1 1]); imshow(matriz,[]); title("Amplitud"); colormap(parula);

[rows, columns,~] = size(matriz);
hold on;
for row = 1:s(1):rows+1
line([0, columns], [row, row], 'Color', 'k', 'LineWidth', 1);
end
for col = 1:s(2):columns+1
line([col, col], [0, rows], 'Color', 'k', 'LineWidth', 1);
end
%frame_h = get(handle(gcf),'JavaFrame');
%set(frame_h,'Maximized',1);
%saveas(gca, "Dataset.svg"); 
% 
% figure;
% pause(0.00001);
% 
% imshow(matriz(:,:,2),[]); title("Fase");colormap(parula);
% [rows, columns,~] = size(matriz);
% hold on;
% for row = 1:s(1):rows+1
% line([0, columns], [row, row], 'Color', 'k', 'LineWidth', 1);
% end
% for col = 1:s(2):columns+1
% line([col, col], [0, rows], 'Color', 'k', 'LineWidth', 1);
% end
% frame_h = get(handle(gcf),'JavaFrame');
% set(frame_h,'Maximized',1);
% saveas(gca, "Fase.svg");