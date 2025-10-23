import sqlite3
from datetime import datetime

DB_NAME = "betika.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        payout REAL,
        bet_amount REAL,
        result TEXT,
        profit_loss REAL
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS status (
        id INTEGER PRIMARY KEY,
        is_running INTEGER
    )''')

    # Ensure only one row for control
    cursor.execute("INSERT OR IGNORE INTO status (id, is_running) VALUES (1, 0)")
    conn.commit()
    conn.close()


def log_bet_result(payout, bet_amount, result, profit_loss):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO bets (timestamp, payout, bet_amount, result, profit_loss)
                      VALUES (?, ?, ?, ?, ?)''',
                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    payout, bet_amount, result, profit_loss))
    conn.commit()
    conn.close()


def set_running(state: bool):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE status SET is_running=? WHERE id=1", (1 if state else 0,))
    conn.commit()
    conn.close()


def is_running():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT is_running FROM status WHERE id=1")
    (status,) = cursor.fetchone()
    conn.close()
    return bool(status)
