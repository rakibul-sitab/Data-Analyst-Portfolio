#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sqlalchemy import create_engine

pip install psycopg2

db_config = {'user': '####',         # user name
             'pwd': '####', # password
             'host': '####',
             'port': ####,              # connection port
             'db': '####'}          # the name of the database

connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'], db_config['pwd'], db_config['host'], db_config['port'], db_config['db'])


# connecting to the database
engine = create_engine(connection_string)

# create an sql query
query = ''' SELECT * FROM trending_by_time'''

# run query and store results in dataframe
trending_by_time = pd.io.sql.read_sql(query, con=engine, index_col = '####')

trending_by_time.to_csv('trending_by_time.csv', index = False)

