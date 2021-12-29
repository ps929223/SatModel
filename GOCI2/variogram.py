'''
https://scikit-gstat.readthedocs.io/en/latest/userguide/variogram.html
'''


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pprint import pprint

## apply the function to a meshgrid and add noise
xx, yy = np.mgrid[0:0.5 * np.pi:500j, 0:0.8 * np.pi:500j]
np.random.seed(42)

## generate a regular field
_field = np.sin(xx)**2 + np.cos(yy)**2 + 10

## add noise
z = _field + np.random.normal(0, 0.15, (500,  500))

plt.imshow(z, cmap='RdYlBu_r')

## Using scikit-gstat
import skgstat as skg

## random cooridnates
np.random.seed(42)
coords = np.random.randint(0, 500, (300, 2))
values = np.fromiter((z[c[0], c[1]] for c in coords), dtype=float)
V = skg.Variogram(coords, values)
V.plot()
V.distance_difference_plot()

''' Distance '''
locations = [[0,0], [0,1], [1,1], [1,0]]
V = skg.Variogram(coordinates=locations, values=[0, 1, 2, 1], model='spherical',
                  dist_func='euclidean', normalize=False)
V.distance

## turn into a 2D matrix again
from scipy.spatial.distance import squareform
print(squareform(V.distance))

help(skg.Variogram)


''' Binning '''
from skgstat.binning import even_width_lags, uniform_count_lags
from scipy.spatial.distance import pdist
loc = np.random.normal(50, 10, size=(30, 2))
distances = pdist(loc)

##  different bin edges for the calculated dummy distance matrix:
even_width_lags(distances, 10, maxlag=250)
uniform_count_lags(distances, 10, maxlag=250)


## Using the Variogram you can see how the setting of different binning methods will
## update the Variogram.bins and eventually n_lags:
test = skg.Variogram(
   *skg.data.pancake().get('sample'),  # use some sample data
   n_lags=25,                          # set 25 classes
   bin_func='even')
print(test.bins)

## sqrt will very likely estimate way more bins
test.bin_func = 'sqrt'
print(f'Auto-derived {test.n_lags} bins.')
print(V.bins)



'''Observation differences'''
coords, vals = skg.data.pancake(N=200).get('sample')
V = skg.Variogram(coords,  vals,  n_lags=25)
V.maxlag = 500

## first 10 distances
V.distance[:10].round(1)

## first 10 groups
V.lag_groups()[:10]

## Actual Variogram Bins
V.bins


'''Experimental variograms'''
fig, _a = plt.subplots(1, 3, figsize=(8,4), sharey=True)
axes = _a.flatten()
axes[0].plot(V.bins, V.experimental, '.b')
V.estimator = 'cressie'
axes[1].plot(V.bins, V.experimental, '.b')
V.estimator = 'dowd'
axes[2].plot(V.bins, V.experimental, '.b')

axes[0].set_ylabel('semivariance')
axes[0].set_title('Matheron')
axes[1].set_title('Cressie-Hawkins')
axes[2].set_title('Dowd')


'The spherical model'

from skgstat import models
# set estimator back
V.estimator = 'matheron'
V.model = 'spherical'
xdata = V.bins
ydata = V.experimental

from scipy.optimize import curve_fit
# initial guess - otherwise lm will not find a range
p0 = [np.mean(xdata), np.mean(ydata), 0]
cof, cov =curve_fit(models.spherical, xdata, ydata, p0=p0)
print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))

xi =np.linspace(xdata[0], xdata[-1], 100)
yi = [models.spherical(h, *cof) for h in xi]
plt.plot(xdata, ydata, 'og', label='bins~experiment')
plt.plot(xi, yi, '-b', label='Spherical')
plt.legend()

####  Trust-Region Reflective (TRF)
# Levenberg-Marquardt is faster and TRF is more robust.
##
V.fit_method ='trf'
V.plot()
pprint(V.parameters)
##
V.fit_method ='lm' # Levenberg-Marquardt
V.plot()
pprint(V.parameters)



''' Exponential model '''
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
axes[0].set_title('Spherical')
axes[1].set_title('Exponential')

V.fit_method = 'trf'

V.plot(axes=axes[0], hist=False)
# switch the model
V.model = 'exponential'
V.plot(axes=axes[1], hist=False)


# spherical
V.model = 'spherical'
rmse_sph = V.rmse
r_sph = V.describe().get('effective_range')

# exponential
V.model = 'exponential'
rmse_exp = V.rmse
r_exp = V.describe().get('effective_range')

print('Spherical   RMSE: %.2f' % rmse_sph)
print('Exponential RMSE: %.2f' % rmse_exp)

print('Spherical effective range:    %.1f' % r_sph)
print('Exponential effective range:  %.1f' % r_exp)


''' Kriging Interpolation '''

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

V.model = 'spherical'
krige1 = V.to_gs_krige()

V.model = 'exponential'
krige2 = V.to_gs_krige()


# build a grid
x = y = np.arange(0, 500, 5)

# apply
field1, _ = krige1.structured((x, y))
field2, _ = krige2.structured((x, y))

# use the same bounds
vmin = np.min((field1, field2))
vmax = np.max((field1, field2))

# plot
axes[0].set_title('Spherical')
axes[1].set_title('Exponential')
axes[0].imshow(field1, origin='lower', cmap='terrain_r', vmin=vmin, vmax=vmax)
axes[1].imshow(field2, origin='lower', cmap='terrain_r', vmin=vmin, vmax=vmax)


# calculate the differences
diff = np.abs(field2 - field1)
print('Mean difference:     %.1f' % np.mean(diff))
print('3rd quartile diffs.: %.1f' % np.percentile(diff, 75))
print('Max differences:     %.1f' % np.max(diff))
plt.imshow(diff, origin='lower', cmap='hot')
plt.colorbar()



''' Defining orientiation '''
from matplotlib.patches import FancyArrowPatch as farrow
fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.arrow(0,0,2,1,color='k')
ax.arrow(-.1,0,3.1,0,color='k')
ax.set_xlim(-.1, 3)
ax.set_ylim(-.1,2.)
ax.scatter([0,2], [0,1], 50, c='r')
ax.annotate('A (0, 0)', (.0, .26), fontsize=14)
ax.annotate('B (2, 1)', (2.05,1.05), fontsize=14)
arrowstyle="Simple,head_width=6,head_length=12,tail_width=1"
ar = farrow([1.5,0], [1.25, 0.625],  color='r', connectionstyle="arc3, rad=.2", arrowstyle=arrowstyle)
ax.add_patch(ar)
ax.annotate('26.5°', (1.5, 0.25), fontsize=14, color='r')


### Calculating orientations
c = np.array([[0,0], [2,1], [1,2], [2, -1], [1, -2]])
east = np.array([1,0])

u = c[1:]   # omit the first one
angles = np.degrees(np.arccos(u.dot(east) / np.sqrt(np.sum(u**2, axis=1))))
angles.round(1)

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.set_xlim(-.1, 2.25)
ax.set_ylim(-2.1,2.1)
ax.arrow(-.1,0,3.1,0,color='k')
for i,p in enumerate(u):
    ax.arrow(0, 0, p[0], p[1], color='r')
    ax.annotate('%.1f°' % angles[i], (p[0] / 2, p[1] / 2), fontsize=14, color='r')
ax.scatter(c[:,0], c[:,1], 50, c='r')

class TestCls(skg.DirectionalVariogram):
    def fit(*args, **kwargs):
        pass

DV = TestCls(c, np.random.normal(0,1,len(c)))
DV._calc_direction_mask_data()
np.degrees(DV._angles + np.pi)[:len(c) - 1]
360 - np.degrees(DV._angles + np.pi)[[2,3]]
DV.tolerance = 90
DV.pair_field()