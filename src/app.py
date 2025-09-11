"""Main Flask application for PhishGuard phishing detection."""

import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest

from .config import Config
from .utils import load_model, predict_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='../templates')
    app.config.from_object(config_class)
    
    # Load model and vectorizer
    try:
        model, vectorizer = load_model(
            app.config['MODEL_PATH'], 
            app.config['VECTORIZER_PATH']
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model, vectorizer = None, None
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """Main page for phishing detection."""
        prediction = None
        confidence = None
        message = ""
        error = None
        
        if request.method == 'POST':
            try:
                message = request.form.get('message', '').strip()
                
                if not message:
                    error = "Please enter a message to analyze."
                elif len(message) > 1000:  # Basic length validation
                    error = "Message is too long. Please keep it under 1000 characters."
                elif model is None or vectorizer is None:
                    error = "Model not available. Please try again later."
                else:
                    # Make prediction
                    pred, conf = predict_message(model, vectorizer, message)
                    prediction = "🛑 Phishing Message" if pred == 1 else "✅ Legitimate Message"
                    confidence = f"{conf:.2%}"
                    
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                error = "An error occurred while analyzing the message. Please try again."
        
        return render_template(
            'index.html', 
            prediction=prediction,
            confidence=confidence,
            message=message,
            error=error
        )
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """API endpoint for programmatic access."""
        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'Message is required'}), 400
            
            message = data['message'].strip()
            if not message:
                return jsonify({'error': 'Message cannot be empty'}), 400
            
            if len(message) > 1000:
                return jsonify({'error': 'Message too long'}), 400
            
            if model is None or vectorizer is None:
                return jsonify({'error': 'Model not available'}), 503
            
            # Make prediction
            pred, conf = predict_message(model, vectorizer, message)
            
            return jsonify({
                'prediction': 'phishing' if pred == 1 else 'legitimate',
                'confidence': conf,
                'is_phishing': bool(pred)
            })
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None and vectorizer is not None
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('index.html', error="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('index.html', error="Internal server error"), 500
    
    return app

def main():
    """Run the application."""
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )

if __name__ == '__main__':
    main()
