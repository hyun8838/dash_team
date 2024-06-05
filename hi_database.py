import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')




def create_table(db_name, csv_file, table_name):
  conn = sqlite3.connect(db_name)
  c = conn.cursor()
  df = pd.read_csv(csv_file)
  df.to_sql(table_name, conn, index=False, if_exists='replace')
  conn.commit()
  conn.close()

# source_table_name: table_name as train, test, costumor_info, data_df, ...

def create_predict_table(db_name, source_table_name, table_name):
  conn = sqlite3.connect(db_name)
  c = conn.cursor()
  # create predict table

  c.execute("PRAGMA table_info(source_table_name)")
  columns_info = c.fetchall()
  columns = [col_info[1] for col_info in columns_info]
  columns_str = ', '.join(columns)

  create_predict_query = f"CREATE TABLE predict({columns_str}, predict TEXT)"
  c.execute(create_predict_query)

  conn.commit()
  conn.close()

def check_tables(db_name):
    """
    Check and print all table names in the database.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    conn.close()
    return [table[0] for table in tables]




def print_table(db_name, table_name):
  conn = sqlite3.connect(db_name)
  c = conn.cursor()

  # print table
  print("\n\n" + table_name + "...")
  query = c.execute("SELECT * FROM " + table_name)
  items = c.fetchall()
  for item in items:
    print(item)

  conn.commit()
  conn.close()


# db에서 데이터 df로 불러오기
def db_to_df(db_name, table_name):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    query = cur.execute( "SELECT * From " + table_name  )
    cols = [column[0] for column in query.description]
    result = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    con.close()
    return result

# df를 db로 저장
def df_to_db(df, db_name, table_name):
  conn = sqlite3.connect(db_name)
  df.to_sql(table_name, conn, if_exists = 'append', index = False)
  conn.close()

# train_test_split
def data_split(df, test_size, random_state):
  train, test = train_test_split(df, test_size, random_state)


def create_pickle(obj, file):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)

def read_pickle(file):
    with open(file, 'rb') as f:
        out = pkl.load(f)
    return out


# 여러 데이터 추가
def df_to_db(dat, db_name, table_name):
  conn = sqlite3.connect(db_name)
  dat.to_sql(table_name, conn, if_exists = 'append', index = False)
  conn.close()



def show_all(db_name, table_name):
  conn = sqlite3.connect(db_name)
  c = conn.cursor()

  c.execute("SELECT rowid, * FROM " + table_name)
  items = c.petchall()

  for item in items:
    print(item)

  conn.commit()
  conn.close()


def call_x_value(db_name, table_name, ind):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query = c.execute("SELECT * FROM " + table_name + " WHERE rowid={}".format(ind))
    cols = [column[0] for column in c.description]
    out = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    conn.close()
    return out

# Usage:
# db_name = "/content/drive/MyDrive/데이터애널리스틱스특수연구3/project/통합/TRAIN.DB"

# check_tables(db_name)

# ecommerce_df = db_to_df(db_name = db_name, table_name = "train_table")
# ecommerce_df
