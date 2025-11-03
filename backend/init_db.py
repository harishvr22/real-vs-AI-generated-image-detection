# backend/init_db.py
import sqlite3
import os

# Ensure the database directory exists
db_path = os.path.join(os.path.dirname(__file__), 'database', 'history.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect and create table
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number TEXT,
    image_name TEXT,
    accuracy REAL,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()

print("✅ Database initialized successfully at:", db_path)
