# PhishGuard 🛡️

A machine learning-powered web application that detects phishing messages using logistic regression and TF-IDF vectorization.

## Features

- **Real-time Detection**: Analyze messages instantly through a web interface
- **Confidence Scores**: Get confidence levels for each prediction
- **REST API**: Programmatic access via JSON API
- **Modern UI**: Clean, responsive interface with visual feedback
- **Error Handling**: Robust error handling and validation
- **Health Monitoring**: Built-in health check endpoint

## Project Structure

```
PhishGuard/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── app.py             # Main Flask application
│   ├── config.py          # Configuration settings
│   ├── training.py        # Model training logic
│   └── utils.py           # Utility functions
├── templates/             # HTML templates
│   └── index.html
├── models/                # Saved models (created after training)
│   ├── phishing_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py                 # Application entry point
├── train_model.py         # Training script
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PhishGuard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python train_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Paste the message you want to analyze in the text area
3. Click "🎯 Check Message" to get the prediction
4. View the result with confidence score

### API Usage

The application provides a REST API for programmatic access:

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
  "message": "Your account has been compromised. Click here to secure it now!"
}
```

**Response**:
```json
{
  "prediction": "phishing",
  "confidence": 0.95,
  "is_phishing": true
}
```

**Health Check**: `GET /health`

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectorization (1000 features)
- **Preprocessing**: Lowercase, remove punctuation and numbers
- **Dataset**: 90 messages (60 phishing, 30 legitimate)
- **Accuracy**: ~90% (varies based on test split)

## Configuration

Environment variables can be used to configure the application:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)
- `SECRET_KEY`: Flask secret key

## Deployment

### Heroku

1. Create a Heroku app
2. Set environment variables if needed
3. Deploy using Git:
   ```bash
   git add .
   git commit -m "Deploy PhishGuard"
   git push heroku main
   ```

### Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 5000
CMD ["python", "app.py"]
```

## Development

### Adding New Training Data

1. Edit `src/training.py`
2. Add new messages to the `create_dataset()` method
3. Retrain the model: `python train_model.py`

### Extending the Model

- Modify `src/training.py` to use different algorithms
- Adjust preprocessing in `src/utils.py`
- Update feature extraction parameters

## API Documentation

### POST /api/predict

Analyze a message for phishing content.

**Request**:
- Content-Type: `application/json`
- Body: `{"message": "string"}`

**Response**:
- 200: Successful prediction
- 400: Bad request (missing/invalid message)
- 503: Model not available

### GET /health

Check application health and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Limitations

- Small training dataset (90 samples)
- May not generalize well to all phishing patterns
- Limited to text-based analysis
- No real-time model updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and demonstration purposes. It should not be the sole method for detecting phishing attempts in production environments.
