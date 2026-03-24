'''
This script is used once to create the SQLite database and its table for storing attendance data.
✅ What it does:
- Creates the attendance.db file (inside database/ folder)
- Creates a table named daily_attendance with the following columns: id, name, date, first_entry, last_exit
'''
import sqlite3  # Library to work with SQLite databases - SQLite is a simple database system
import os  # Library for file and folder operations
os.makedirs('database', exist_ok=True) # Create database folder if it doesn't exist, (exist_ok=True means "it's okay if it exists")
conn = sqlite3.connect('database/attendance.db') # If attendance.db doesn't exist, it creates a new database file. If attendance.db already exists, it connects to it
cursor = conn.cursor() # Create a cursor object to run SQL commands. A cursor is like a pointer that helps us execute commands in the database. You need this cursor to create tables, insert data, or read data.
# Create the 'daily_attendance' table to store name, date, only first entry and last exit
cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_attendance (
        name TEXT NOT NULL,                    -- Person's name (required field)
        date TEXT NOT NULL,                    -- Date (YYYY-MM-DD format, required field)
        first_entry TEXT,                      -- First entry time (HH:MM:SS format, optional)
        last_exit TEXT                         -- Last exit time (HH:MM:SS format, optional)
    )
''')
conn.commit()  # Save all changes to the database file
conn.close()   # Close the database connection
print("✅ daily_attendance table created.") # Print success message to let user know everything worked

'''
IMPORTANT NOTES:
1. Run this script only once when setting up the system for the first time
2. If you run it again, it won't create duplicate tables (because of IF NOT EXISTS)
3. This creates table for storing all attendance records
4. The database file will be created at: database/attendance.db
'''