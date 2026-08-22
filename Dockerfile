# ParamsCalibrator microservice: calibrates motor-winding thermal constants
# (hA, k_wh) from a window of sensor data via the trained MLP (or, for
# comparison, any of the five classical baselines).
#
# The image builds its own training data and trains its own model at build
# time (RUN step below) rather than copying in pre-built artifacts, so the
# service is fully reproducible from source alone -- no external data, no
# proprietary Siemens material, nothing to hand-carry into the image.
#
#   docker build -t paramscalibrator .
#   docker run -p 8000:8000 paramscalibrator
#   curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Generate the synthetic datasets and train both MLP calibrators inside the
# image. --no-window-sweep skips the (informative but non-essential)
# accuracy-vs-window-length sweep, since only the trained model artifacts
# under models/ are needed at runtime.
RUN python scripts/generate_datasets.py \
    && python scripts/train_ml.py --no-window-sweep

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
