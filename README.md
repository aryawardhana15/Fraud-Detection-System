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

## ❓ **FAQ & Troubleshooting**
- **Kafka/MongoDB tidak jalan?** Pastikan Docker Desktop sudah running, lalu cek dengan `docker ps`.
- **Tidak bisa connect ke Kafka?** Pastikan port `9092` terbuka dan service Kafka sudah up.
- **Error Python module?** Install dependensi dengan `pip install ...` sesuai kebutuhan.

