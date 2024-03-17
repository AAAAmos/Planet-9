import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

import lmfit
from lmfit.lineshapes import gaussian2d
from lmfit.models import LorentzianModel

def gaussian(x, amp, cen, sig):
    """
    1-d gaussian: gaussian(x, amp, cen, wid)
    
    input: number, array
    """
    return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2))

def gaussian2d(x, y, amp, cenx, ceny, sigx, sigy):
    """
    2-d gaussian: gaussian(x, y, amp, cenx, ceny, sigx, sigy)
    
    input: number, array
    """
    return (amp / (2*np.pi * sigx*sigy)) * np.exp(-((x-cenx)**2 / (2*sigx**2)) - ((y-ceny)**2 / (2*sigy**2)))

def CrossmatchDisfit(file, cname, fitrange=70, grid=101, weight=1, dim=2):

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600
    
    if dim == 1:
        
        fig, ax = plt.subplots()
        
        data = {
            'ra': ax.hist(x, grid)[1],
            'rac': ax.hist(x, grid+1)[0],
            'dec': ax.hist(y, grid)[1],
            'decc': ax.hist(y, grid+1)[0]
        }
        
        plt.close(fig)
        
        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()
        
        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)

        dframe = pd.DataFrame(data=data)

        model = LorentzianModel()

        paramsx = model.guess(dframe['rac'], x=dframe['ra'])
        paramsy = model.guess(dframe['decc'], x=dframe['dec'])

        resultra = model.fit(dframe['rac'], paramsx, x=dframe['ra'])
        cen1x = resultra.values['center']
        sig1x = resultra.values['sigma']
        resultdec = model.fit(dframe['decc'], paramsy, x=dframe['dec'])
        cen1y = resultdec.values['center']
        sig1y = resultdec.values['sigma']
        
        fitx = model.func(dframe['ra'], **resultra.best_values)
        fity = model.func(dframe['dec'], **resultdec.best_values)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        ax = axs[0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ell = Ellipse(
                (cen1x, cen1y),
                width = 3*sig1x,
                height = 3*sig1y,
                edgecolor = 'w',
                facecolor = 'none'
            )
        ax.add_patch(ell)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('Delta RA [arcsec]')
        ax.set_ylabel('Delta DEC [arcsec]')

        ax = axs[1]
        ax.plot(dframe['ra'], dframe['rac'], marker='.', ls='')
        ax.plot(dframe['ra'], fitx)
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1x, sig1x))
        ax.set_xlabel('Delta RA [arcsec]')
        ax.set_ylabel('count')

        ax = axs[2]
        ax.plot(dframe['dec'], dframe['decc'], marker='.', ls='')
        ax.plot(dframe['dec'], fity)
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1y, sig1y))
        ax.set_xlabel('Delta DEC [arcsec]')
        ax.set_ylabel('count')
        
        fig.suptitle('list3 x '+file.split('-')[1]+'  1D fitting')
        
        plt.show()
    
    if dim == 2:

        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        
        w = z**weight+0.1
        
        model = lmfit.models.Gaussian2dModel()
        params = model.guess(z, xf, yf)
        result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
        Amp = result.values['amplitude']
        cenx = result.values['centerx']
        sigx = result.values['sigmax']
        ceny = result.values['centery']
        sigy = result.values['sigmay']
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)
        
        fit = model.func(X, Y, **result.best_values)

        Zx = Z[int((grid+1)/2)]
        fitx = fit[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]
        fity = fit.T[int((grid+1)/2)]

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        plt.rcParams.update({'font.size': 15})
        # plt.rcParams.update({"tick.labelsize": 13})
        
        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ell = Ellipse(
                (cenx, ceny),
                width = 3*sigx,
                height = 3*sigy,
                edgecolor = 'w',
                facecolor = 'none'
            )
        ax.add_patch(ell)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[0, 1]
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ax.set_title('Residual')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[1, 0]
        ax.plot(xedges[:100], Zx, marker='.', ls='')
        ax.plot(xedges[:100], fitx)
        ax.set_title('Center:{0:5.4f}, 1σ:{1:5.3f}'.format(cenx, sigx))
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[1, 1]
        ax.plot(yedges[:100], Zy, marker='.', ls='')
        ax.plot(yedges[:100], fity)
        ax.set_title('Center:{0:5.4f}, 1σ:{1:5.3f}'.format(ceny, sigy))
        ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6]+'  2D fitting')

        plt.show()

def CrossGaussian2dfit(file, cname, fitrange=60, grid=100, weight=0.9, fitonly=False, sqslice=False):

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])
    
    if sqslice:
        data = data.loc[(x>-0.02)&(x<0.02)&(y>-0.02)&(y<0.02),
                    ['RA', 'DEC', n1, n2]
                    ]

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600

    xedges = np.linspace(-fitrange, fitrange, grid)
    yedges = np.linspace(-fitrange, fitrange, grid)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    H = H.T

    z = H.flatten()

    X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
    xf = X.flatten()
    yf = Y.flatten()
    # error = np.sqrt(z+1)
    w = z**weight+0.1

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, xf, yf)
    result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)

    if not fitonly:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)

        vmax = np.nanpercentile(Z, 99.9)

        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Histogram of Data')

        ax = axs[0, 1]
        fit = model.func(X, Y, **result.best_values)
        art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Fit')

        ax = axs[1, 0]
        fit = model.func(X, Y, **result.best_values)
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Data - Fit')

        ax = axs[1, 1]
        ax.scatter(
            x, y,
            s = 1,
            alpha=0.2
        )
        ax.set_title('Origin data points from'+file.split('-')[1])

        for ax in axs.ravel():
            ax.set_xlabel('Delta RA [arcsec]')
            ax.set_ylabel('Delta DEC [arcsec]')
        
        plt.show()

    if fitonly:
        
        return result

def Errorbar(x, y, yerr, c='black', elinewidth=1, alpha=1):
    
    '''
    The plt.errorbar has problem on the oerder of layer, so I am making this function
    x: np.array
    y: np.array
    yerr: np.array
    
    This is a garbige, exaggerate zorder
    '''
    
    # fig, ax = plt.subplots()
    
    for i in range(len(x)):
        
        ymax1 = y[i]+yerr[i]
        ymin1 = y[i]-yerr[i]
        # print(ymax1, ymin1)
        plt.vlines(x=x[i], ymin=ymin1, ymax=ymax1, colors=c, linewidth=elinewidth, alpha=alpha)