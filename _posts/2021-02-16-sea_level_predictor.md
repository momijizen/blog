---
title: "The Sea-Level Change Prediction"
categories:
  - Project
tags:
  - linregress
---

[Link: Notebook](https://github.com/momijizen/boilerplate-sea-level-predictor/blob/main/sea_level_predictor.ipynb)

># The Sea-Level Change Prediction

>We will analyze the global average sea-level change dataset since 1880 and use the data to forecast sea-level change through the year 2050.

>We import the dataset used from https://datahub.io/core/sea-level-rise

Import libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
!pip -q install datapackage
from datapackage import Package
```

Download Dataset


```python
package = Package('https://datahub.io/core/sea-level-rise/datapackage.json')
# print list of all resources:
print(package.resource_names)
```

    ['validation_report', 'csiro_alt_gmsl_mo_2015_csv', 'csiro_alt_gmsl_yr_2015_csv', 'csiro_recons_gmsl_mo_2015_csv', 'csiro_recons_gmsl_yr_2015_csv', 'epa-sea-level_csv', 'csiro_alt_gmsl_mo_2015_json', 'csiro_alt_gmsl_yr_2015_json', 'csiro_recons_gmsl_mo_2015_json', 'csiro_recons_gmsl_yr_2015_json', 'epa-sea-level_json', 'sea-level-rise_zip', 'csiro_alt_gmsl_mo_2015', 'csiro_alt_gmsl_yr_2015', 'csiro_recons_gmsl_mo_2015', 'csiro_recons_gmsl_yr_2015', 'epa-sea-level']


 Global Average Absolute Sea Level Change, 1880-2014


```python
# to load only epa-sea-level_csv dataset
resources = package.resources
resource = resources[5]
df = pd.read_csv(resource.descriptor['path'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>CSIRO Adjusted Sea Level</th>
      <th>Lower Error Bound</th>
      <th>Upper Error Bound</th>
      <th>NOAA Adjusted Sea Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1880-03-15</td>
      <td>0.000000</td>
      <td>-0.952756</td>
      <td>0.952756</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1881-03-15</td>
      <td>0.220472</td>
      <td>-0.732283</td>
      <td>1.173228</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1882-03-15</td>
      <td>-0.440945</td>
      <td>-1.346457</td>
      <td>0.464567</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1883-03-15</td>
      <td>-0.232283</td>
      <td>-1.129921</td>
      <td>0.665354</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1884-03-15</td>
      <td>0.590551</td>
      <td>-0.283465</td>
      <td>1.464567</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2010-03-15</td>
      <td>8.901575</td>
      <td>8.618110</td>
      <td>9.185039</td>
      <td>8.122973</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2011-03-15</td>
      <td>8.964567</td>
      <td>8.661417</td>
      <td>9.267717</td>
      <td>8.053065</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2012-03-15</td>
      <td>9.326772</td>
      <td>8.992126</td>
      <td>9.661417</td>
      <td>8.457058</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2013-03-15</td>
      <td>8.980315</td>
      <td>8.622047</td>
      <td>9.338583</td>
      <td>8.546648</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2014-03-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.663700</td>
    </tr>
  </tbody>
</table>
<p>135 rows × 5 columns</p>
</div>



Prepare Data


```python
df = df[['Year','CSIRO Adjusted Sea Level']]
```


```python
df = df.dropna(axis=0)
```


```python
df['Year'] = df['Year'].str.slice(0,4)
df = df.astype({'Year': 'int32'})
df.set_index('Year', inplace=True)
```

Create scatter plot


```python
# Create scatter plot
plt.figure(figsize=(14,6))
plt.scatter(df.index, df['CSIRO Adjusted Sea Level'], alpha=0.5 )

# Create first line of best fit
res = linregress(df.index, df['CSIRO Adjusted Sea Level'])
year_1880_2050 = np.concatenate((df.index, np.arange(2014,2050)), axis=0)
plt.plot(year_1880_2050 , res.intercept + res.slope*year_1880_2050, 'r', label='fitted line 1')

# Create second line of best fit
res2 = linregress(df.loc['200':].index, df.loc['200':]['CSIRO Adjusted Sea Level'])
year_2000_2050 = np.concatenate((df.loc['200':].index, np.arange(2014,2050)), axis=0)
plt.plot(year_2000_2050 , res2.intercept + res2.slope*year_2000_2050, 'g', label='fitted line 2')

 # Add labels and title
plt.xlabel('Year')
plt.ylabel('Sea Level (inches)')
plt.title('Rise in Sea Level')
plt.legend()

```


![png](https://raw.githubusercontent.com/momijizen/blog/master/assets/img_sea_level/output_13_1.png)
