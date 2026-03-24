'''
This script generates a formatted attendance report from the SQLite database.
It reads attendance data and displays it in a nice table format on the screen.
'''
import sqlite3  # Import sqlite3 for database operations - this connects to and reads from the database
from tabulate import tabulate  # Import tabulate for creating formatted tables - this makes data look nice in columns
# Configuration: Database file path
DB_PATH = 'database/attendance.db'  # Path to the SQLite database file - ensure this matches your DB path
'''
Function to fetch attendance records from the database.
This function connects to the database, retrieves all attendance data, and returns it.
HOW IT WORKS:
1. Opens a connection to the SQLite database
2. Executes a query to get all attendance records
3. Sorts the results by date and then by name
4. Returns all the records as a list
'''
def fetch_attendance_records():
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)  # Create connection to the SQLite database
    cursor = conn.cursor()  # Create cursor object to execute SQL commands
    cursor.execute("SELECT name, date, first_entry, last_exit FROM daily_attendance ORDER BY date, name")  # Fetch data from the updated table: daily_attendance where the Results are sorted by date first, then by name alphabetically.
    records = cursor.fetchall()  # Get all records returned by the query
    conn.close()  # Close the database connection to free up resources
    return records  # Return the list of attendance records
'''
Main execution block - runs when the script is executed directly.
This section handles displaying the attendance report to the user.
'''
if __name__ == "__main__":  # If this script is run directly (not imported)
    data = fetch_attendance_records()  # Get all attendance records from database
    '''
    Display the attendance data in a formatted table.
    If there's data, create a nice table. If not, show a message.
    '''
    if data:  # If there are attendance records in the database
        headers = ["Name", "Date", "First Entry", "Last Exit"]  # Define column headers for the table
        print("\n📋 Attendance Report (First Entry & Last Exit per Day):\n")  # Print report title
        print(tabulate(data, headers=headers, tablefmt="grid"))  # Display data in a grid table format
    else:  # If no attendance records found
        print("⚠️ No attendance records found in database.")  # Print message indicating no data