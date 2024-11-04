import matplotlib.pyplot as plt
import numpy as np
import rasterio
import matplotlib.colors as colors
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

fig, axs = plt.subplots(nrows=9, ncols=4, figsize=(15.75, 21), gridspec_kw={'wspace': 0.05, 'hspace': 0.2}, sharex=True, sharey=True)
locations = ['florida', 'florida', 'florida', 'minnesota', 'minnesota', 'minnesota', 'california', 'california', 'california']
models = ['self', 'gate', 'simple', 'se']
titles = ['ASPAM-U-Net', 'AG-U-Net', 'SE-U-Net', 'Simple-U-Net']
mae_values = {}
mape_values = {}
rmse_values = {}
#
for j in range(9):
    max_val = 0
    min_val = 1000000000

    for i in range(4):
        if j % 3 == 0:
            metric = 'mape'
        elif j % 3 == 1:
            metric = 'mae'
        else:
            metric = 'rmse'

        img = rasterio.open(
            f'D:/results of predicitons/tif files/{locations[j]}_first half of year_{models[i]}_{metric}.tif')
        arr = img.read()
        arr = arr[:, :832, :1600]
        print(arr.shape,'shape')
        if metric == 'mae':
            arr = arr
        if metric == 'rmse':
            arr = arr

        flattened_arr = arr.flatten()
        flattened_arr.sort()
        index_98_percentile = np.percentile(flattened_arr, 99)
        max_val_98_percentile = flattened_arr[flattened_arr <= index_98_percentile].max()
        max_val = max(max_val, max_val_98_percentile)
        min_val = min(min_val, arr.min())
        print(max_val,min_val)
    for i in range(4):
        if j % 3 == 0:
            metric = 'mape'
        elif j % 3 == 1:
            metric = 'mae'
        else:
            metric = 'rmse'

        img = rasterio.open(
            f'D:/results of predicitons/tif files/{locations[j]}_first half of year_{models[i]}_{metric}.tif')
        arr = img.read()
        arr = arr[:, :832, :1600]
        if metric == 'mae':
            arr = arr
        if metric == 'rmse':
            arr = arr

        if metric == 'mape':
            mape_values[(locations[j], models[i])] = round(np.mean(arr), 4)
        elif metric == 'mae':
            mae_values[(locations[j], models[i])] = round(np.mean(arr), 4)
        else:
            rmse_values[(locations[j], models[i])] = round(np.mean(arr), 4)

        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        axs[j, i].set_xticks([])
        axs[j, i].set_yticks([])
        im = axs[j, i].imshow(np.squeeze(arr), cmap='RdYlBu_r', norm=norm)

        if j == 0:
            axs[j, i].set_title(titles[i], fontsize=12,weight='bold',y=1.02)
        if i ==0:
            if j%3==0:
                axs[j, i].set_ylabel(f'MAPE (%)', fontsize=12,)
            elif j%3==1:
                axs[j, i].set_ylabel(f'MAE (mm)', fontsize=12,)
            else:
                axs[j, i].set_ylabel(f'RMSE (mm)', fontsize=12, )

        if i == 3:
            cax = fig.add_axes([0.905, 0.81-j * 0.0875, 0.006, 0.07], label=f"colorbar{j}")
            fig.colorbar(im, cax=cax, shrink=0.7, pad=0.05)
            cax.yaxis.tick_right()
            cax.yaxis.set_label_position('right')
            cax.tick_params(axis='y', labelsize=7.5)
            cb_shape = FancyBboxPatch((0, 0), 1, 1, boxstyle='round', edgecolor='none', )
            cax.add_patch(cb_shape)
    if j == 2:
        axs[j, 1].annotate('Study Area 1', xy=(1.027, -0.083), xycoords='axes fraction',
                           ha='center', va='center', fontsize=11, fontweight='bold')
    elif j == 5:
        axs[j, 1].annotate('Study Area 2', xy=(1.027, -0.083), xycoords='axes fraction',
                           ha='center', va='center', fontsize=11, fontweight='bold')
    elif j == 8:
        axs[j, 1].annotate('Study Area 3', xy=(1.027, -0.083), xycoords='axes fraction',
                           ha='center', va='center', fontsize=11, fontweight='bold')

print("MAE values:", mae_values)
print("MAPE values:", mape_values)
print("RMSE values:", rmse_values)
plt.savefig('C:/Users/user/Desktop/my_plot_error_summer.png', dpi=250, bbox_inches='tight')
plt.show()
