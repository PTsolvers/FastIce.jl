clear

do_save = 0;

no_dat    = -9999;
path_data = '../data';
name_bed  = 'B73-12_GlacierBed';
name_surf = 'B73-12_SwissALTI3D_r2019';

tif      = Tiff([path_data, '/', name_bed, '.tif'],'r');
img_bed  = read(tif);
tif      = Tiff([path_data, '/', name_surf, '.tif'],'r');
img_surf = read(tif);

bed  = img_bed;  bed2  = bed; 
surf = img_surf; surf2 = surf;


bed2(bed<(no_dat+1)) = NaN;

surf2(bed<(no_dat+1)) = no_dat;
surf2(surf2<(no_dat+1)) = NaN;

ele = surf2 - bed2;

% crop
ystart = 168;
xend   = 319;
bed_dat  = double(bed2(1:xend,ystart:end));
surf_dat = double(surf2(1:xend,ystart:end));
ele_dat  = double(ele(1:xend,ystart:end));


figure(1),clf
subplot(311), pcolor(bed_dat'), shading flat, axis equal tight, colorbar, title('bedrock elevation')
subplot(312), pcolor(surf_dat'), shading flat, axis equal tight, colorbar, title('surface elevation')
subplot(313), pcolor(ele_dat'), shading flat, axis equal tight, colorbar, title('ice thickness')

if do_save==1, save([path_data, '/', 'arolla3D.mat'], 'bed_dat','surf_dat','ele_dat'); end
