from __future__ import print_function
import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

# Reindexing allows values not in the original DataFrame values.
# It fills in these rows with NaN values
# This allows using indexing with an external list without having to sanitizing the input
print(cities.reindex([0, 4, 5, 2]))
