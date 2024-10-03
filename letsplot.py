import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
import seaborn as sns

import re
def get_element_ratio(composition):
    elements = {'Cs': 'A', 'MA': 'A', 'FA': 'A', 'Rb': 'A', 'Pb': 'B', 'Sn': 'B', 'I': 'X', 'Br': 'X', 'Cl': 'X'}
    elements_is_mul = {key: 0 for key in elements.keys()}
    element_ratio = {key: 0 for key in elements.keys()}

# Separate the elements in the chemical formula
    mul_factors = {key: [] for key in elements.keys()}
    if ')' in composition:
        pattern2 = r"\(([^()]+)\)"
        match = re.findall(pattern2, composition)
        for m in match:
            for ele in elements.keys():
                if ele in m:
                    temp_str = composition.split('(' + m + ')')[1]
                    pattern3 = r'([0-9]*\.?[0-9]*)'
                    pattern4 = r'([A-Za-z]+)'
                    elements_is_mul[ele] = 1

                    value = elements[ele]
                    mul_fact = re.findall(pattern3,temp_str)[0]
                    mul_factors[ele].append(float(mul_fact))

# Traverse each element and calculate its proportion
    for ele in elements.keys():
        if ele in composition:

            pattern3 = r'([0-9]*\.?[0-9]*)'
            if ele == 'Pb' and len(composition.split(ele)) >2:
                element_ratio['Pb'] = []
                for temp_str in composition.split(ele)[1:3]:
                    if not re.match('\d',temp_str):
                        element_ratio[ele].append(1)
                    else:
                        element_ratio[ele].append(float(re.findall(pattern3,temp_str)[0]))
                # pass
            else:

                temp_str = composition.split(ele)[1]
                if not re.match('\d',temp_str):
                    element_ratio[ele] = 1
                else:
                    element_ratio[ele] = float(re.findall(pattern3,temp_str)[0])

    # If the element exists in the dictionary, calculate its proportion
    for symbol, position in elements.items():
        if position == 'A':
            if len(mul_factors[symbol])>0:
                element_ratio[symbol] =  mul_factors[symbol][0] * element_ratio[symbol]
        elif position == 'B':
            if len(mul_factors[symbol])==1:
                element_ratio[symbol] =  mul_factors[symbol][0] * element_ratio[symbol]
            elif len(mul_factors[symbol])==2:
                element_ratio[symbol] =  mul_factors[symbol][0] * element_ratio[symbol][0] + mul_factors[symbol][1] * element_ratio[symbol][1]
        else:
            element_ratio[symbol] /= 3.0
            if len(mul_factors[symbol])>0:
                element_ratio[symbol] =  mul_factors[symbol][0] * element_ratio[symbol]

    return element_ratio

def calc_corr(a,b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
    return corr_factor

from sklearn.metrics import mean_squared_error

def letsplot(train,trainpre,test,testpre,modelname = 'ML',target = 'PCE'):
    fontsize = 12
    plt.figure(figsize=(3,3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"

    a = plt.scatter(train, trainpre, s=25,c='#b2df8a')
    plt.plot([train.min(), train.max()], [train.min(),train.max()], 'k:', lw=1.5)
    plt.xlabel('PCE Observation', fontsize=fontsize)
    plt.ylabel('PCE Prediction', fontsize=fontsize)
    plt.tick_params(direction='in')
    plt.title('{} model for {} prediction'.format(modelname,target),fontsize=fontsize)

    b = plt.scatter(test, testpre, s=25,c='#4682B4')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    plt.show()

    print ('Train r:',calc_corr(train, trainpre))
    print ('Train R2:',r2_score(train, trainpre))
    print ('Train RMSE:', np.sqrt(metrics.mean_squared_error(train, trainpre)))
    print('--------------------------------------')
    print ('Test r:', calc_corr(test, testpre))
    print ('Test R2:',r2_score(test, testpre))
    print ('Test RMSE:', np.sqrt(metrics.mean_squared_error(test, testpre)))

# Discovering Outliers Based on Model Predictive Functions
def find_outliers(model, X, y, sigma=2):
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # Calculate the parameter z defined by the outlier, where data with | z | greater than σ will be considered abnormal
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    print('R2 = ', model.score(X, y))
    print('MSE = ', mean_squared_error(y, y_pred))
    print('------------------------------------------')
    print('mean of residuals', mean_resid)
    print('std of residuals', std_resid)
    print('------------------------------------------')
    print(f'find {len(outliers)}', 'outliers： ')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))

    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('z')

    return outliers

def getdata(y_train, y_train_hat,y_test, y_test_hat):
    data_train = np.column_stack((y_train, y_train_hat))
    data_test = np.column_stack((y_test, y_test_hat))
    data = pd.DataFrame(np.append(data_train, data_test, axis=0), columns=["Observation", "Prediction"])
    data['type'] = ['Train' if i < len(data_train) else 'Test' for i in range(len(data))] # 添加类型列
    return data


def myscatterplot(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 8,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # plot_height = 6
    plot_aspect = 1.2
    plot_palette = ["#BC3C28", "#0072B5"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .6
    xlim_left = -0.1
    xlim_right = 26
    ylim_bottom = -0.1
    ylim_top = 26

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k", 'family': 'Times New Roman'}
    text_fontdict = {'family': 'Times New Roman', 'size': '22', 'weight': 'bold', 'color': 'black'}

    fig, ax = plt.subplots(figsize=(plot_height, plot_height),dpi=300)

    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.65,
                    edgecolor=plot_scatter_kw.get('edgecolor', 'none'), palette=plot_palette, ax=ax)

    ax.set_facecolor(face_color)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(spine_color)
    ax.tick_params(labelsize=label_size, direction=direction)  # 修改坐标轴数字大小

    ax.grid(which=grid_which, ls=grid_ls, c=grid_c, alpha=grid_alpha)

    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    ax.set_title(f"{modelname} for {target} prediction", fontdict=title_fontdict)

    ax.set_xlabel(x_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Times New Roman'})
    ax.set_ylabel(y_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Times New Roman'})
    ax.plot([-0.5, 25.5], [-0.5, 25.5], linestyle='--', color='gray', linewidth=2)

    sns.set(font_scale=1.5)  
    plt.legend(loc='upper left', fontsize=16)  

    train_text1 = 'Train R: {:.4f}'.format(calc_corr(y_train, y_train_hat))
    test_text1 = 'Test R: {:.4f}'.format(calc_corr(y_test, y_test_hat))
    test_text2 = 'Test R2: {:.4f}'.format(r2_score(y_test, y_test_hat))

    ax.text(0.67, 0.25, train_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.19, test_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.13, test_text2, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    # Test RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_hat))
    rmse_text = 'Test RMSE: {:.3f}'.format(test_rmse)
    ax.text(0.67, 0.07, rmse_text, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    if savepic is True:
        plt.savefig('./img/{}.png'.format(picname),bbox_inches = 'tight',transparent = True) #,transparent = True
    else:
        pass
    print('Train R2:',r2_score(y_train, y_train_hat))
    print('Train RMSE:',np.sqrt(mean_squared_error(y_train, y_train_hat)))
    print('Train MAE:',mean_absolute_error(y_train, y_train_hat))
    print('Test MAE:',mean_absolute_error(y_test, y_test_hat))
    plt.show()

import pandas as pd
import numpy as np

def save_plot_data(y_train, y_train_hat, y_test, y_test_hat, savename):
    data = {'y_train': y_train,
            'y_train_predict': y_train_hat,
            'y_test': y_test,
            'y_test_predict': y_test_hat}
    df = pd.DataFrame(data)
    df.to_csv('./img/{}.csv'.format(savename), index=False)

import pandas as pd
import numpy as np

def save_arrays_with_nan(y_train, y_train_hat, y_test, y_test_hat, savename):

    max_length = max(len(y_train), len(y_train_hat), len(y_test), len(y_test_hat))
    
    filled_y_train = np.pad(y_train, (0, max_length - len(y_train)), mode='constant', constant_values=np.nan)
    filled_y_train_hat = np.pad(y_train_hat, (0, max_length - len(y_train_hat)), mode='constant', constant_values=np.nan)
    filled_y_test = np.pad(y_test, (0, max_length - len(y_test)), mode='constant', constant_values=np.nan)
    filled_y_test_hat = np.pad(y_test_hat, (0, max_length - len(y_test_hat)), mode='constant', constant_values=np.nan)
    
    df = pd.DataFrame({'y_train': filled_y_train,
                       'y_train_predict': filled_y_train_hat,
                       'y_test': filled_y_test,
                       'y_test_predict': filled_y_test_hat})
    df.to_csv('./img/{}.csv'.format(savename), index=False)
