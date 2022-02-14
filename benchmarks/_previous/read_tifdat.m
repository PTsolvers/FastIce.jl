clear

do_save = 1;

no_dat    = -9999;
path_data = '../data';
name_bed  = 'B73-12_GlacierBed';
name_surf = 'B73-12_SwissALTI3D_r2019';

tif   = Tiff([path_data, '/', name_bed, '.tif'],'r');
zbed  = read(tif);
tif   = Tiff([path_data, '/', name_surf, '.tif'],'r');
zsurf = read(tif);

% crop
ystart  = 168;
yend    = 360;
xend    = 280;
max_alt = 2960;
zbed(zbed>max_alt)=no_dat;
zbed    = double(zbed(1:xend,ystart:yend));
zsurf   = double(zsurf(1:xend,ystart:yend));

% preprocess
mask   = abs(1-(zbed==no_dat));
zthick = zsurf - zbed;
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

% if do_save==1, save([path_data, '/', 'arolla3D.mat'], 'zbed', 'zsurf', 'mask', 'xv', 'yv'); end

% xv2   = mask.*xv2;
% yv2   = mask.*yv2;
% zavg  = mask.*zavg;

% LSQ fit
xv2_  = xv2;  xv2_(mask==0)=[];
yv2_  = yv2;  yv2_(mask==0)=[];
zavg_ = zavg; zavg_(mask==0)=[];

A = [xv2_(:), yv2_(:), ones(size(xv2_(:)))];
B = zavg_(:);

x = (A'*A)\(A'*B);

plane = xv2*x(1) + yv2*x(2) + x(3);

%%
zbed(mask == 0) = plane(mask == 0);
mask2 = mask; mask2([1,end],:)=1; mask2(:,[1,end])=1;
[x,y] = ndgrid(xv,yv);
x     = x(:);
y     = y(:);
zbed  = zbed(:);
mask2  = mask2(:);
x(mask2 == 0) = [];
y(mask2 == 0) = [];
zbed(mask2 == 0) = [];

[x2,y2] = ndgrid(linspace(min(x),max(x),280),linspace(min(y),max(y),193));
zbed    = griddata(x,y,zbed,x2,y2);
%%

if do_save==1, save([path_data, '/', 'arolla3D_2.mat'], 'zbed', 'zsurf', 'mask', 'plane', 'xv', 'yv'); end

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
