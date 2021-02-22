# Import MySQL Connector Driver
import mysql.connector as mysql

# Load the credentials from the secured .env file
import os
import xlrd
from dotenv import load_dotenv
load_dotenv('credentials.env')

db_user = os.environ['MYSQL_USER']
db_pass = os.environ['MYSQL_PASSWORD']
db_name = os.environ['MYSQL_DATABASE']
#db_host = os.environ['MYSQL_HOST'] # must 'localhost' when running this script outside of Docker
db_host = 'localhost'

# Connect to the database
db = mysql.connect(user=db_user, password=db_pass, host=db_host, database=db_name)
cursor = db.cursor()

# # CAUTION!!! CAUTION!!! CAUTION!!! CAUTION!!! CAUTION!!! CAUTION!!! CAUTION!!!
cursor.execute("drop table if exists Users;")
cursor.execute("drop table if exists Vehical_Population_Default;")
cursor.execute("drop table if exists Mileage_Default;")
cursor.execute("drop table if exists FE_Numbers_Default;")
cursor.execute("drop table if exists EV_Numbers_Default;")

# Create a TStudents table (wrapping it in a try-except is good practice)
try:
  cursor.execute("""
    CREATE TABLE Users (
      id          integer  AUTO_INCREMENT PRIMARY KEY,
      first_name  VARCHAR(30) NOT NULL,
      last_name   VARCHAR(30) NOT NULL,
      email       VARCHAR(50) NOT NULL,
      password    VARCHAR(20) NOT NULL,
      created_at  TIMESTAMP
    );
  """)
except:
  print("Users table already exists. Not recreating it.")
  
try:
  cursor.execute("""
    CREATE TABLE Vehical_Population_Default (
      id          integer  AUTO_INCREMENT PRIMARY KEY,
      time_period  VARCHAR(30),
      Passenger_Vehicle    DOUBLE,
      Taxi       VARCHAR(50) NOT NULL,
      password    VARCHAR(20) NOT NULL,
      created_at  TIMESTAMP
    );
  """)
except:
  print("Users table already exists. Not recreating it.")

# Insert Records
query = "insert into Users (first_name, last_name, email, password, created_at) values (%s, %s, %s, %s, %s)"
values = [
  ('rick','gessner','rick@gessner.com', 'abc123', '2020-02-20 12:00:00'),
  ('ramsin','khoshabeh','ramsin@khoshabeh.com', 'abc123', '2020-02-20 12:00:00'),
  ('al','pisano','al@pisano.com', 'abc123', '2020-02-20 12:00:00'),
  ('truong','nguyen','truong@nguyen.com', 'abc123', '2020-02-20 12:00:00')
]
cursor.executemany(query, values)
db.commit()

# Selecting Records
cursor.execute("select * from Users;")
print('---------- DATABASE INITIALIZED ----------')
[print(x) for x in cursor]
db.close()
