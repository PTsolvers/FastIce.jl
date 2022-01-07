clear

load('../out_visu/out_pa.mat')
load('../out_visu/out_res.mat')

[xc2,yc2] = ndgrid(xc,yc);

xc2r = xc2*cos(al) - yc2*sin(al);
yc2r = xc2*sin(al) + yc2*cos(al);

yc2r = yc2r-min(yc2r(:));


Pt_v  = Pt;  Pt_v(Phase~=1)=NaN;
Vn_v  = Vn;  Vn_v(Phase~=1)=NaN;
tII_v = zeros(size(Pt));
tII_v(2:end-1,2:end-1) = tII; tII_v(Phase~=1)=NaN;

figure(1),clf

subplot(311), pcolor(xc2r, yc2r, Pt_v),shading flat,colorbar,title('Pressure')
subplot(312), pcolor(xc2r, yc2r, Vn_v),shading flat,colorbar, title('||V||')
subplot(313), pcolor(xc2r, yc2r, tII_v),shading flat,colorbar, title('\tau_{II}')

print('-dpng','-r200','../figs/arolla.png')
