from flask import Flask, render_template, request, jsonify
from aviator_script import start_bot, stop_bot, is_running, log_messages
import csv, os
from database import init_db


init_db()
app = Flask(__name__)

@app.route('/')
def dashboard():

    return render_template("dashboard.html")

@app.route('/start', methods=['POST'])
def start():
    data = request.get_json()
    bet_amount = float(data.get("bet_amount", 1.2))
    phone = data.get("phone")
    password = data.get("password")
    check_interval = int(data.get("check_interval", 15))
    check_duration = int(data.get("check_duration", 2))

    resp = start_bot(bet_amount, phone, password, check_interval, check_duration)
    return jsonify({"status": resp})

@app.route('/stop', methods=['POST'])
def stop():
    resp = stop_bot()
    return jsonify({"status": resp})

@app.route('/status')
def status():
    return jsonify({"running": is_running()})

@app.route("/history")
def history():
    path = "bet_results.csv"
    if not os.path.exists(path):
        return jsonify([])

    data = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                data.append({
                    "round": row[0],
                    "payout": row[1],
                    "timestamp": row[2] if len(row) > 2 else ""
                })
    return jsonify(data[-20:])  # last 20

@app.route("/logs")
def get_logs():
    """Return the latest runtime logs."""
    return jsonify(log_messages[-10:])  # last 50 logs

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
