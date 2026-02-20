FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY harmonized_food_prices.csv .
COPY templates/ templates/
COPY static/ static/

# Copy pre-trained models if available (optional)
COPY models/ models/ 2>/dev/null || true

EXPOSE 5000

CMD ["python", "app.py"]
