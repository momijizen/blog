##UTM (Universal Transverse Mercator) coordinates


```python
df_raw[['name_thai','coordinates_x','coordinates_y']].head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name_thai</th>
      <th>coordinates_x</th>
      <th>coordinates_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>สถานีหมอชิต</td>
      <td>667955.316838</td>
      <td>1.526435e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>สถานีสะพานควาย</td>
      <td>667518.538626</td>
      <td>1.525461e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>สถานีอารีย์</td>
      <td>666980.056603</td>
      <td>1.523898e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>สถานีสนามเป้า</td>
      <td>666707.778693</td>
      <td>1.523115e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>สถานีอนุเสาวรีย์ชัย</td>
      <td>666173.478204</td>
      <td>1.522018e+06</td>
    </tr>
  </tbody>
</table>
</div>



##Convert a UTM coordinate into latitude and longitude coordinates


```python
!pip install -q django  
from django.contrib.gis.geos import Polygon 
from pprint import pprint
```


```python
coor = (df_raw[['coordinates_x','coordinates_y']]).to_numpy().tolist() 
#Polygon first and last coordinate should be identical (linear ring)
coor.append(coor[0]) 

poly_thai = Polygon(coor, srid=32647)
poly_gps = poly_thai.transform(4326, clone=True) #4326
#pprint(poly_gps.coords)
```


```python
df_lat_long = pd.DataFrame(poly_gps[0], columns=['longitude','latitude'])
df_lat_long.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.553787</td>
      <td>13.802568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100.549689</td>
      <td>13.793794</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.544616</td>
      <td>13.779691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.542052</td>
      <td>13.772630</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.537046</td>
      <td>13.762744</td>
    </tr>
  </tbody>
</table>
</div>



###Merge latitude and longitude columns


```python
df_station = df_raw.merge(df_lat_long, left_index=True, right_index=True)
df_station[['name_thai','coordinates_x','coordinates_y','longitude','latitude']].head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name_thai</th>
      <th>coordinates_x</th>
      <th>coordinates_y</th>
      <th>longitude</th>
      <th>latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>สถานีหมอชิต</td>
      <td>667955.316838</td>
      <td>1.526435e+06</td>
      <td>100.553787</td>
      <td>13.802568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>สถานีสะพานควาย</td>
      <td>667518.538626</td>
      <td>1.525461e+06</td>
      <td>100.549689</td>
      <td>13.793794</td>
    </tr>
    <tr>
      <th>2</th>
      <td>สถานีอารีย์</td>
      <td>666980.056603</td>
      <td>1.523898e+06</td>
      <td>100.544616</td>
      <td>13.779691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>สถานีสนามเป้า</td>
      <td>666707.778693</td>
      <td>1.523115e+06</td>
      <td>100.542052</td>
      <td>13.772630</td>
    </tr>
    <tr>
      <th>4</th>
      <td>สถานีอนุเสาวรีย์ชัย</td>
      <td>666173.478204</td>
      <td>1.522018e+06</td>
      <td>100.537046</td>
      <td>13.762744</td>
    </tr>
  </tbody>
</table>
</div>


