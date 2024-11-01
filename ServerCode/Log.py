def Save(data):
    import sqlite3

    # Define the log data
    logdata = data

    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('Log.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create the table if it doesn't already exist
    cursor.execute('''
    INSERT INTO loggie (timestamp, item, Param1, Param2, Image) 
    VALUES (?, ?, ?, ?, ?)
    ''', logdata)


    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Log data inserted successfully.")

def Getall():
    import sqlite3

    # Connect to the database
    conn = sqlite3.connect('Log.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Select everything from the loggie table
    cursor.execute('SELECT * FROM loggie')

    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    # Print the retrieved rows

    # Close the connection
    conn.close()
    return rows
def Start():
    import sqlite3

    conn = sqlite3.connect('Log.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create the table if it doesn't already exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS loggie (
        timestamp TEXT,
        item TEXT,
        Param1 TEXT,
        Param2 TEXT,
        Image TEXT
    )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Created Successfully...")
