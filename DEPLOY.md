# Deploying the Demand Forecasting UI

This guide covers **three ways** to deploy the prediction app so it runs on any computer.

---

## Option 1: Run Locally (Simplest)

### Prerequisites

- Python 3.9+ installed
- The `harmonized_food_prices.csv` file in the project root

### Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open in browser
# http://127.0.0.1:5000
```

The app will train models from the CSV on first boot (~60 seconds), then serve the UI.

If you've already run the notebook on Colab and downloaded the `models/` folder, place it in the project root — the app will load those pre-trained champion models instead (instant startup).

---

## Option 2: Docker (Works on Any Machine)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### Steps

```bash
# 1. Build the image
docker build -t demandcast .

# 2. Run the container
docker run -p 5000:5000 demandcast

# 3. Open in browser
# http://localhost:5000
```

### Share with others

```bash
# Save the image to a file
docker save demandcast -o demandcast.tar

# On another machine, load and run
docker load -i demandcast.tar
docker run -p 5000:5000 demandcast
```

---

## Option 3: Deploy to the Cloud (Render — free tier)

[Render](https://render.com) offers free hosting for Python web services.

### Steps

1. **Push your project to GitHub** (make sure `harmonized_food_prices.csv` is included or uploaded separately)

2. **Create a Render account** at https://render.com

3. **New Web Service** → Connect your GitHub repository

4. **Configure:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3.11

5. **Deploy** — Render will build and host your app with a public URL

> **Note:** On Render's free tier, the app may take ~60s to cold-start (training models). For faster startup, commit the `models/` directory from your Colab run.

---

## Project Files Required for Deployment

| File / Folder                | Purpose                          |
| ---------------------------- | -------------------------------- |
| `app.py`                     | Flask backend + prediction API   |
| `requirements.txt`           | Python dependencies              |
| `templates/index.html`       | Frontend HTML                    |
| `static/css/style.css`       | Styling                          |
| `static/js/app.js`           | Frontend logic                   |
| `harmonized_food_prices.csv` | Training data                    |
| `models/` _(optional)_       | Pre-trained models from notebook |
| `Dockerfile` _(optional)_    | For Docker deployment            |
