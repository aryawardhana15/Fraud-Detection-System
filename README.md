# FinShield Link - Fraud Detection System

Sistem deteksi fraud transaksi real-time berbasis Kafka, AI (LSTM), dan MongoDB. Project ini mensimulasikan aliran transaksi, mendeteksi pola mencurigakan, dan menyimpan alert ke database. Cocok untuk demo, hackathon, atau riset fintech!

---

## 🚀 **Arsitektur Sistem**

```
[Simulasi Transaksi]
        |
        v
     [Kafka]
        |
        v
[Deteksi Fraud (AI)]
        |
        v
    [MongoDB]
        |
        v
   [Dashboard]
```

- **Kafka**: Message queue untuk transaksi real-time
- **MongoDB**: Menyimpan alert transaksi mencurigakan
- **LSTM Model**: AI untuk deteksi pola fraud
- **Dashboard**: (Opsional) Monitoring alert di browser

---

## 📦 **Struktur File**

- `data_ingest_py.py`  &rarr; Simulasi & streaming transaksi ke Kafka
- `risk_train_py.py`   &rarr; Training model AI (LSTM) deteksi fraud
- `risk_model_py.py`   &rarr; Deteksi fraud real-time dari Kafka ke MongoDB
- `docker-compose.yml` &rarr; Menyalakan Kafka, Zookeeper, MongoDB via Docker

---

## 🛠 **Instalasi & Persiapan**

1. **Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)** dan pastikan sudah berjalan.
2. **Clone/download** project ini ke komputer Anda.
3. **Jalankan infrastruktur** (Kafka, MongoDB):
   ```bash
   docker compose up -d
   ```
4. **Install dependensi Python** (opsional, jika belum):
   ```bash
   pip install kafka-python torch scikit-learn joblib numpy pandas
   ```

---

## ▶️ **Cara Menjalankan Sistem**

### 1. **Simulasi Transaksi ke Kafka**
```bash
python data_ingest_py.py
```

### 2. **Training Model AI (LSTM)**
```bash
python risk_train_py.py
```

### 3. **Deteksi Fraud Real-Time**
```bash
python risk_model_py.py
```

### 4. **(Opsional) Dashboard Monitoring**
Jika Anda ingin dashboard web, silakan buat dengan Flask yang membaca data dari MongoDB.

---

## 🧠 **Penjelasan Cara Kerja**

1. **Simulasi transaksi**: Script membuat data transaksi (normal & fraud) dan mengirim ke Kafka.
2. **Training AI**: Model LSTM dilatih dari data simulasi untuk mengenali pola fraud.
3. **Deteksi real-time**: Script membaca transaksi dari Kafka, prediksi fraud, dan simpan alert ke MongoDB.
4. **Dashboard**: (Opsional) Menampilkan alert dari MongoDB di browser.

---

## ⚙️ **Kustomisasi & Pengembangan**
- Anda bisa mengubah logika simulasi di `data_ingest_py.py`.
- Model AI bisa diganti/ditingkatkan di `risk_train_py.py` dan `risk_model_py.py`.
- Threshold deteksi fraud bisa diatur sesuai kebutuhan.
- Dashboard bisa dikembangkan dengan Flask, React, dsb.

---

## ❓ **FAQ & Troubleshooting**
- **Kafka/MongoDB tidak jalan?** Pastikan Docker Desktop sudah running, lalu cek dengan `docker ps`.
- **Tidak bisa connect ke Kafka?** Pastikan port `9092` terbuka dan service Kafka sudah up.
- **Error Python module?** Install dependensi dengan `pip install ...` sesuai kebutuhan.

