'''
This script synchronizes attendance data from a CSV file to a SQLite database.
It reads the CSV file (like an Excel spreadsheet) and copies the data into a database.

WHAT THIS SCRIPT DOES:
1. Reads attendance data from a CSV file
2. Connects to a SQLite database
3. Updates or inserts attendance records in the database
4. Ensures both CSV and database have the same information

WHY THIS IS NEEDED because: If we make any changes to the csv file, then we can run this code and ensure same data is updated in the database. 
'''
import sqlite3  # Import sqlite3 for database operations - this handles the database connection
import csv  # Import csv for reading CSV files - this reads Excel-like files
import os  # Import os for file system operations - this checks if files exist

CSV_FILE = 'attendance_log.csv'  # Path to the CSV file containing attendance data
DB_PATH = 'database/attendance.db'  # Path to the SQLite database file
'''
Main function that synchronizes CSV data to the database.
This function does all the work of copying data from CSV to database.

HOW IT WORKS:
1. Checks if both CSV file and database exist
2. Reads all data from the CSV file
3. For each person's attendance record, updates the database
4. Handles both new records and updates to existing records
'''
def sync_csv_to_db():
    if not os.path.exists(CSV_FILE):  # Checks if the CSV file doesn't exist
        print(f"❌ CSV file not found: {CSV_FILE}")  # Print error message
        return  # Exit the function
    if not os.path.exists(DB_PATH):  # Check if the database file doesn't exist
        print(f"❌ Database file not found: {DB_PATH}")  # Print error message
        return  # Exit the function
    conn = sqlite3.connect(DB_PATH)  # Create connection to the SQLite database
    cursor = conn.cursor()  # Create cursor for executing SQL commands
    '''
    Read all data from the CSV file into memory.
    This loads the entire attendance spreadsheet so we can process it.
    '''
    with open(CSV_FILE, 'r') as f:  # Open the CSV file for reading
        reader = csv.reader(f)  # Create CSV reader object
        rows = list(reader)  # Convert all rows into a list
    '''
    Validate the CSV format.
    The CSV must have at least 3 columns: Name, Date, and at least one appearance.
    '''
    if not rows or len(rows[0]) < 3:  # If no rows are there in the list variable 'rows' or header has less than 3 columns
        print("❌ Invalid CSV format. Header must be: Name, Date, <at least one time>")  # Print error
        return  # Exit the function
    '''
    Process each row of attendance data (skip the header row).
    For each person's daily attendance, extract their name, date, and Appearances...
    '''
    for row in rows[1:]:  # Loop through all rows except the header (first row)
        if len(row) < 3:  # If row doesn't have minimum required columns
            continue  # Skip this row as the data is incomplete and go to the next one
        name = row[0].strip()  # Get person's name and remove leading and trailing whitespace spaces
        date = row[1].strip()  # Get date and remove extra spaces
        times = [t.strip() for t in row[2:] if t.strip()]  # Get all non-empty time entries after skipping first two columns i.e., name and date. if t.strip(): checks is this value non-empty after removing spaces? But it does not change t itself, t is still the original unstripped string. So we use t.strip() to remove leading/trailing spaces or newlines.
        # Skip rows with missing essential data
        if not name or not date or not times:  # If name, date, or times are missing like if we have name = ''(after strip) , date = ''(after strip) and times = []  (nothing left after stripping). this passes the condition of less than 3 columns becuase it has atleast 3 columns but they are empty.
            continue  # Skip this row and go to the next one
        '''
        Extract first appearance time and last appearance time from all recorded times.
        - first appearance time: The first time the person entered that day
        -last appearance time: The last time the person left that day
        '''
        first_entry = times[0]  # First time in the list is the first entry
        last_exit = times[-1]  # Last time in the list is the last exit
        '''
        Update if record exists, insert if it doesn't.
        This ensures we don't create duplicate records for the same person and date.
        '''
        # Check if a record already exists for this person and date
        # COUNT(*) helps check whether the database already contains a row with the same name and date as the current row from the CSV file.
        cursor.execute("SELECT COUNT(*) FROM daily_attendance WHERE name=? AND date=?", (name, date))  # Execute SQL query to count existing records. '?' is a parameter placeholder used in parameterized queries. Instead of directly putting values like: WHERE name='Abhay' AND date='2025-07-14' we use ? and pass the values separately.
        exists = cursor.fetchone()[0]  # fetchone() returns a tuple. Suppose it returns (2,) → meaning 2 rows matched. Now, fetchone()[0] just extracts that number (2)
        if exists:  # this is same as if exists != 0. If record already exists in database then update it
            cursor.execute("UPDATE daily_attendance SET first_entry=?, last_exit=? WHERE name=? AND date=?", (first_entry, last_exit, name, date))  # Updates all the existing matched record with new times, even if more than 1 row is matched at once.
        else:  # If record doesn't exist in database
            cursor.execute("INSERT INTO daily_attendance (name, date, first_entry, last_exit) VALUES (?, ?, ?, ?) """, (name, date, first_entry, last_exit))  # Insert new record
    conn.commit()  # Save all changes to the database
    conn.close()  # Close the database connection
    print("✅ Synced CSV data to SQLite database successfully.")  # Print success message
if __name__ == "__main__":  # To make sure that this script runs when executed directly (not when imported)
    sync_csv_to_db()  # Call the main synchronization function