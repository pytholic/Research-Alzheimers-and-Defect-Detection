x0 = csvread('D:/LAB Project Data/1D signal stuff/airforce/Data051116_163928_oven/1D_new/nodefect/point(57,175).csv');
t = 0:1:500;
x = interp1(x0(:,1),t,'linear');

N = length(x);
S = zeros(N, N);

for i = 1:N,
    S(:,i) = abs( repmat( x(i), N, 1 ) - x(:) );
end

imagesc(t, t, S)
axis square