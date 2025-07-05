from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Ganti di production

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET'])
@login_required
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

@app.route('/user/<user_id>', methods=['GET'])
@login_required
def user_history(user_id):
    # Ambil semua transaksi/alert user dari MongoDB
    alerts = list(db_alerts.find({'user_id': user_id}).sort('timestamp', -1))
    for alert in alerts:
        if '_id' in alert and not isinstance(alert['_id'], str):
            alert['_id'] = str(alert['_id'])
    return render_template('index.html', alerts=alerts, user_id=user_id, start_date='', end_date='')

@app.route('/profile/<user_id>')
@login_required
def user_profile(user_id):
    alerts = list(db_alerts.find({'user_id': user_id}).sort('timestamp', -1))
    total_transaksi = len(alerts)
    total_alert = sum(1 for a in alerts if a.get('fraud_score', 0) >= 0.7)
    avg_amount = sum(a.get('amount', 0) for a in alerts) / total_transaksi if total_transaksi else 0
    avg_behavior = sum(a.get('user_behavior_score', 0) for a in alerts) / total_transaksi if total_transaksi else 0
    return render_template('user_profile.html', user_id=user_id, alerts=alerts, total_transaksi=total_transaksi, total_alert=total_alert, avg_amount=avg_amount, avg_behavior=avg_behavior)

if __name__ == '__main__':
    app.run(debug=True) 