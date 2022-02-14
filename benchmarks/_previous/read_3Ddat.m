clear

do_save = 1;

path_data = '../data';
name_bed   = 'BedrockElevation_ArollaCrop';
name_thick = 'IceThickness_ArollaCrop';

tif   = Tiff([path_data, '/', name_bed, '.tif'],'r');
zbed  = read(tif);
tif   = Tiff([path_data, '/', name_thick, '.tif'],'r');
zthick = read(tif);

zbed(zbed<0)    = NaN;
zthick(zthick<=0) = 0;

% crop
ystart  = 125;
yend    = 300;
xstart  = 220;
xend    = 480;
max_alt = 3010;
nsmo    = 2;

zthick(zbed>max_alt) = 0;
zbed    = double( zbed(xstart:xend,ystart:yend));
zthick  = double(zthick(xstart:xend,ystart:yend));
zthick(1:140,80:end) = 0;

if sum(sum(isnan(zbed))) > 0, error('NaNs'); end

for ism=1:nsmo
    zthick(2:end-1,2:end-1) = zthick(2:end-1,2:end-1) + 1/4.1*(diff(zthick(:,2:end-1),2,1) + diff(zthick(2:end-1,:),2,2));
end

zsurf = zbed + zthick;

figure(1),clf
subplot(211),imagesc(zbed'), axis xy, colorbar
subplot(212),imagesc(zthick'), axis xy, colorbar

% preprocess
mask = ones(size(zthick));
mask(zthick==0) = 0; 
zavg   = 0.5.*(zsurf + zbed);

dx  = 10;
dy  = 10;
nxv = size(zbed,1);
nyv = size(zbed,2);
Lx  = dx*(nxv-1);
Ly  = dy*(nyv-1);
xv  = 0:dx:Lx;
yv  = 0:dy:Ly;
[xv2,yv2] = ndgrid(xv,yv);

if do_save==1, save([path_data, '/', 'arolla3D.mat'], 'zbed', 'zsurf', 'zthick', 'mask', 'xv', 'yv'); end

% LSQ fit
xv2_  = xv2;  xv2_(mask==0)=[];
yv2_  = yv2;  yv2_(mask==0)=[];
zavg_ = zavg; zavg_(mask==0)=[];
A = [xv2_(:), yv2_(:), ones(size(xv2_(:)))];
B = zavg_(:);
x = (A'*A)\(A'*B);
plane = xv2*x(1) + yv2*x(2) + x(3);

% rotate
zbedr  = (zbed  - plane).*mask;
zsurfr = (zsurf - plane).*mask;
zmin   = min(zbedr(:));
zbedr  = zbedr  - zmin;
zsurfr = zsurfr - zmin;

% preprocess
zmin   = min(zbedr(:));
zbedr  = zbedr  - zmin;
zsurfr = zsurfr - zmin;
maxDz  = max((zsurf(:) - zbed(:)).*mask(:)); % redundant
lz     = max(zsurfr(:));
lx     = max(xv) - min(xv);
ly     = max(yv) - min(yv);
xc     = 0.5*(xv(1:end-1)+xv(2:end));
yc     = 0.5*(yv(1:end-1)+yv(2:end));
nx     = length(xc);
ny     = length(yc);
resz   = ceil(lz/lx*nx);
% resz   = resz > tx ? resz : tx
% shiftz = resz % tz
nz     = resz; % fact_ny * (shifty < ty/2 ? Int(resy - shifty) : Int(resy + ty - shifty)) - olen
nzv    = nz+1;
zv     = linspace(0,lz,nzv);
zc     = 0.5*(zv(1:end-1)+zv(2:end));

% visu
xv2_v = xv2; xv2_v(mask==0)=NaN;
yv2_v = yv2; yv2_v(mask==0)=NaN;
zbed_v   = zbed;   zbed_v(mask==0)=NaN;
zsurf_v  = zsurf;  zsurf_v(mask==0)=NaN;
zthick_v = zthick; zthick_v(mask==0)=NaN;
zavg_v   = zavg;   zavg_v(mask==0)=NaN;
plane_v  = plane;  plane_v(mask==0)=NaN;
zbedr_v  = zbedr;  zbedr_v(mask==0)=NaN;
zsurfr_v = zsurfr; zsurfr_v(mask==0)=NaN;

figure(1),clf
subplot(211)
scatter3(xv2_v(:), yv2_v(:), zavg_v(:), 6, zavg_v(:), 'filled')
hold on, scatter3(xv2_v(:), yv2_v(:), plane(:),'filled'), hold off
subplot(212)
scatter3(xv2_v(:), yv2_v(:), zsurfr_v(:), 6, zsurfr_v(:), 'filled')
hold on, scatter3(xv2_v(:), yv2_v(:), zbedr_v(:), 6, zbedr_v(:), 'filled'), hold off


figure(2),clf
subplot(311), pcolor(zbed_v'), shading flat, axis equal tight, colorbar, title('bedrock elevation')
subplot(312), pcolor(zsurf_v'), shading flat, axis equal tight, colorbar, title('surface elevation')
subplot(313), pcolor(zthick_v'), shading flat, axis equal tight, colorbar, title('ice thickness')
