import sqlite3
import schedule
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

test_db = "TEST.DB"
train_db = "TRAIN.DB"

def csv2db():
  train_df = pd.read_csv("train.csv")
  test_df = pd.read_csv("test.csv")

  train_conn = sqlite3.connect(train_db)
  test_conn = sqlite3.connect(test_db)

  train_df.to_sql("train_table", train_conn, if_exists="replace", index=False)
  test_df.to_sql("test_table", test_conn, if_exists="replace", index=False)

  print("csv to db")
  train_conn.commit()
  test_conn.commit()
  train_conn.close()
  test_conn.close()

def shoot_row():
  train_conn = sqlite3.connect(train_db)
  test_conn = sqlite3.connect(test_db)

  try:
    train_cur = train_conn.cursor()
    test_cur = test_conn.cursor()

    # Get the maximum ID from train_table
    train_cur.execute("SELECT MAX(IND) FROM train_table")
    last_index = train_cur.fetchone()[0]

    next_index = last_index + 1
    test_cur.execute(f"SELECT * FROM test_table WHERE IND = {next_index}")
    row = test_cur.fetchone()

    if row:
      col_names = [description[0] for description in test_cur.description]
      placeholders = ', '.join(['?'] * len(col_names))
      train_cur.execute(f"INSERT INTO train_table ({', '.join(col_names)}) VALUES ({placeholders})", row)
      train_conn.commit()
      print(f"Row transferred from TEST.DB to TRAIN.DB successfully.")
  # except sqlite3.Error as e:
  #   print(f"An Error Occurred: {e}")

  finally:
    train_conn.close()
    test_conn.close()

# 2 + 1 초마다 shoot
def main():
  csv2db()
  schedule.every(2).seconds.do(shoot_row)

  while True:
    schedule.run_pending()
    time.sleep(1)


if __name__ == "__main__":
  main()