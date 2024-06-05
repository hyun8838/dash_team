#!/usr/bin/env python
# coding: utf-8

# In[1]:

import csv
import sqlite3
import pandas as pd
import numpy as np

class DatabaseManager:
    def __init__(self, db_path='./sy_db.db'):
        self.db_path = db_path

    def set_database(self, db_path):
        self.db_path = db_path
        print(f"Database path set to: {self.db_path}")

    def create_database(self, db_path):
        con = sqlite3.connect(db_path)
        con.close()
        print(f"Database created at: {db_path}")
        self.set_database(db_path)

    def create_table(self, data_name, data_table_name):
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        # Drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {data_table_name}")
        print(f"\n\nExisting table '{data_table_name}' dropped, and new table is created")
        
        # Read data from CSV and create a new table
        data = pd.read_csv(f"./{data_name}.csv")
        data.to_sql(data_table_name, con, index=False, if_exists='replace')

        con.commit()
        con.close()

    def making_dataframe(self, table_name):
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()

        cursor.execute(f"SELECT * FROM {table_name}")
        cols = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(data=cursor.fetchall(), columns=cols)
        
        con.commit()
        con.close()
        return df

    def create_new_table(self, data_frame, data_table_name):
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        # Drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {data_table_name}")
        print(f"\n\nExisting table '{data_table_name}' dropped, and new table is created")
        
        # Create a new table and insert data from the DataFrame
        data_frame.to_sql(data_table_name, con, index=False, if_exists='replace')
        
        con.commit()
        con.close()



# In[ ]:




