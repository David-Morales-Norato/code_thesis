close all; clear; clc; 
test = load('mnist_test.csv');
images = test(1:64,2:785); clear test;
images = images'/255;
images = reshape(images, [28, 28, 64]);
images = permute(images, [2 1 3]);
angle_x = imresize(images(:,:,55), [128 128]);
angle_x = angle_x./(max(angle_x(:)));
mask = double(angle_x>=0.2);
I = mask.*(angle_x).*255;
R = I; G = I; B = I; 

Image = image(I, 'AlphaData', mask);
xticks([]);
yticks([]);

