from flask import Flask, render_template, request
from pymongo import MongoClient
from datetime import datetime, timedelta

app = Flask(__name__)

# Koneksi ke MongoDB (default: localhost:27017)
client = MongoClient('mongodb://localhost:27017/')
db = client['finsheild']  # Ganti sesuai nama database Anda jika berbeda
db_alerts = db['alerts']  # Ganti sesuai nama collection Anda jika berbeda

# Custom filter untuk konversi timestamp ke ICT
@app.template_filter('to_datetime_ict')
def to_datetime_ict_filter(value):
    try:
        # Parsing ISO format
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        # Tambah 7 jam untuk ICT
        dt_ict = dt + timedelta(hours=7)
        return dt_ict.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return value

@app.route('/', methods=['GET'])
def index():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    query = {}
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            query['timestamp'] = {'$gte': start_dt.isoformat()}
        except:
            pass
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            if 'timestamp' in query:
                query['timestamp']['$lt'] = end_dt.isoformat()
            else:
                query['timestamp'] = {'$lt': end_dt.isoformat()}
        except:
            pass
    if query:
        alerts = list(db_alerts.find(query).sort('timestamp', -1))
    else:
        alerts = list(db_alerts.find().sort('timestamp', -1).limit(100))
    # Fix: Convert ObjectId to string for JSON serialization
    for alert in alerts:
        if '_id' in alert and not isinstance(alert['_id'], str):
            alert['_id'] = str(alert['_id'])
    return render_template('index.html', alerts=alerts, start_date=start_date or '', end_date=end_date or '')

if __name__ == '__main__':
    app.run(debug=True) 