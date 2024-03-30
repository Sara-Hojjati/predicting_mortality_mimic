#%%
import psycopg2
import subprocess
pg_ctl_path = "pg_ctl"
data_dir = r"C:\Users\sarho66\AppData\Local\anaconda3\envs\mimic-iii\data\postgres"
start_command = [pg_ctl_path, 'start', '-D', data_dir]

try:
    # Execute the command to start PostgreSQL
    subprocess.run(start_command, check=True, text=True)
    print("PostgreSQL server started successfully.")
except subprocess.CalledProcessError as e:
    print(f"Failed to start PostgreSQL server: {e}")

# Check PostgreSQL server status using Windows service command (if PostgreSQL is installed as a service)
# This example assumes PostgreSQL service is named 'postgresql-x64-<version>'
service_name = "postgresql-x64-<version>"  # Replace <version> with your specific version
status_command = ['sc', 'query', service_name]

try:
    # Execute the command to check PostgreSQL service status
    result = subprocess.run(status_command, check=True, text=True, capture_output=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Failed to check PostgreSQL server status: {e}")

# Connect to the database using psycopg2.connect() function
try:
    conn = psycopg2.connect(dbname="mimic", user="sarho66", password="Ayasakvnkeanavom1994", host="localhost")
    print("Connected successfully")
except Exception as e:
    print(f"Connection failed: {e}")

###


schema_name = 'mimiciii'
cur = conn.cursor() #a cursor allows for executing postgresql commands in python
cur.execute('SET search_path to ' + schema_name) #specifying where to look for the schema
#%%