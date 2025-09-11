"""Training module for PhishGuard phishing detection model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import MODEL_FILE, VECTORIZER_FILE
from .utils import preprocess_text

logger = logging.getLogger(__name__)

class PhishGuardTrainer:
    """Trainer class for PhishGuard phishing detection model."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.training_data = None
        
    def create_dataset(self) -> pd.DataFrame:
        """Create the training dataset with phishing and legitimate messages."""
        
        # Original dataset with 30 phishing and 30 legitimate messages
        original_data = {
            'message': [
                # Phishing Messages (1)
                "Your account has been compromised. Click here to secure it now!",
                "Congratulations! You won a $1000 gift card. Claim now!",
                "This is an official update from your bank. Login to verify your account.",
                "Reminder: Your subscription will be canceled unless you update payment details.",
                "Urgent! Your PayPal account has been locked. Click to verify.",
                "Free money waiting for you! Sign up now.",
                "Your Netflix payment has failed. Update your card to continue service.",
                "We detected unusual activity on your account. Secure it now!",
                "Your bank account has been locked. Verify immediately.",
                "You have an unpaid invoice. Pay now to avoid legal action!",
                "Confirm your password within 24 hours to prevent deactivation.",
                "New message from HR: Your paycheck is delayed. Enter details here.",
                "Security alert: Someone accessed your account from an unknown device.",
                "Government grant available for you! Claim your funds now!",
                "Limited-time offer! Click here to receive your special discount.",
                "Amazon order confirmation: Your package is delayed, update payment.",
                "Apple ID security breach detected. Verify now!",
                "Your social security number has been compromised! Call us now.",
                "Win a free iPad! Just enter your details here.",
                "Your insurance policy is expiring. Renew now to stay covered.",
                "Get rich quick! Invest in this new crypto scheme today.",
                "A hacker has accessed your device. Pay now to remove the threat!",
                "Your email storage is full. Click here to upgrade for free.",
                "Final warning: Your account will be deleted in 24 hours.",
                "Confirm your phone number to receive a special reward.",
                "You have received a private message from the CEO. Open now!",
                "New friend request waiting for you! Click here to view.",
                "Netflix free trial extended! Click to renew your subscription.",
                "Congratulations! Your loan request has been pre-approved!",
                "Dear customer, your credit card has been blocked. Reactivate now!",

                # Legitimate Messages (0)
                "Hey, how have you been? Let's catch up soon!",
                "Meeting at 5 PM. Let me know if you need changes.",
                "Reminder: Your doctor's appointment is at 10 AM tomorrow.",
                "Let's grab coffee tomorrow morning. What time works for you?",
                "The package you ordered has been shipped and will arrive soon.",
                "Team meeting at 3 PM. Don't forget to join the call.",
                "Can you review the document I sent and let me know your thoughts?",
                "Your flight itinerary is confirmed. Safe travels!",
                "Dinner at my place this weekend? Let me know if you're free!",
                "Hey! Just wanted to check in and see how you're doing.",
                "Don't forget your gym session at 6 PM today!",
                "Your library books are due next week. Please return them on time.",
                "Weather alert: Heavy rain expected tomorrow. Drive safely!",
                "Congratulations on your work anniversary! Keep up the great work.",
                "The team outing is scheduled for next Friday. Excited to see you all!",
                "Your subscription renewal was successful. Thank you for staying with us.",
                "Weekend plans? Let's go hiking on Saturday!",
                "Grandma says hi and hopes to see you soon!",
                "Reminder: Rent payment is due next Monday.",
                "The conference call has been moved to 2 PM. Update your calendar.",
                "New software update available. Click here to learn more.",
                "Don't forget to pick up groceries on your way home!",
                "Movie night at 8 PM! Bring some snacks.",
                "Hope you're having a great day! Let's talk soon.",
                "I left my jacket at your place. Can you bring it tomorrow?",
                "Congrats on your promotion! Let's celebrate soon.",
                "The office will be closed on Friday for a company holiday.",
                "Reminder: Your phone bill is due in three days.",
                "Family dinner on Sunday night. Hope you can make it!",
                "The training session starts at 9 AM. Be on time!",
            ],
            'label': [1] * 30 + [0] * 30
        }

        # Additional phishing messages
        additional_phishing = [
            "Your Amazon account is on hold. Click here to verify your billing details.",
            "Final warning: Your account will be terminated in 24 hours unless verified.",
            "Get paid instantly! Apply now for our easy loan approval.",
            "We couldn't process your payment. Click here to update your billing info.",
            "Win a free cruise trip! Just enter your details to claim your prize.",
            "You've exceeded your email quota. Click here to increase your storage.",
            "Official notice: IRS refund available. Submit your info to claim.",
            "Verify your mobile number to prevent service interruption.",
            "Unauthorized login detected. Secure your account immediately.",
            "Claim your $100 Walmart gift card now. Limited-time offer.",
            "Your Dropbox account will be deactivated. Save your files now!",
            "Suspicious activity found. Log in to protect your identity.",
            "You have unpaid taxes. Pay now to avoid penalties.",
            "Free crypto giveaway! Don't miss your chance.",
            "Your eBay transaction failed. Re-enter payment details.",
            "Security update required. Install the latest patch here.",
            "Bank alert: New beneficiary added. Approve or dispute now.",
            "Click to activate your security upgrade.",
            "You are eligible for debt relief. Apply before midnight.",
            "Subscription failed. Update info to continue service.",
            "Download your invoice now to avoid late fees.",
            "Unlock your device by verifying your Apple ID.",
            "You've received a confidential voicemail. Listen now.",
            "RE: Contract issue – open the attached document.",
            "Final notice from IT department – reset your password.",
            "Unusual charges detected. See your activity here.",
            "Your FedEx delivery is on hold. Update your address.",
            "Get exclusive access to new stocks. Register today.",
            "Confirm your email address to receive your refund.",
            "Unlock your bonus funds by logging in now."
        ]

        # Additional legitimate messages
        additional_legitimate = [
            "Hey, did you check the group chat today?",
            "Your car service is scheduled for tomorrow at 11 AM.",
            "Just wanted to say thanks for your help earlier!",
            "I'll be late for the meeting — stuck in traffic.",
            "The budget report is ready. Let me know what you think.",
            "Can you send me the notes from yesterday's class?",
            "Let's reschedule lunch — how's Friday?",
            "We've pushed the deployment to next week.",
            "Happy birthday! Hope you have an amazing day.",
            "Here's the link to our vacation photos!",
            "Dinner was great — we should do it again soon.",
            "I've shared the doc with you. Please review by Friday.",
            "Good luck on your interview tomorrow!",
            "Mom said dinner is at 7 tonight.",
            "Don't forget to bring your ID for the flight.",
            "Looking forward to the weekend getaway!",
            "Can you help me move next weekend?",
            "Let me know when you're free for a quick call.",
            "The tickets are booked for Saturday night.",
            "FYI: The AC is being fixed on Monday morning.",
            "We're planning a surprise party — shhh!",
            "Zoom link for the call: [link]",
            "Coffee on me next time :)",
            "Just got home — super tired!",
            "Is the team still meeting in the big room?",
            "Gym tonight? I could use the motivation.",
            "Did you see the new episode? It's wild.",
            "Here's the recipe you asked for!",
            "Package arrived! Left it on your desk.",
            "Family group chat blowing up again lol."
        ]

        # Combine all data
        all_messages = (original_data['message'] + 
                       additional_phishing + 
                       additional_legitimate)
        all_labels = (original_data['label'] + 
                     [1] * len(additional_phishing) + 
                     [0] * len(additional_legitimate))

        return pd.DataFrame({
            'message': all_messages,
            'label': all_labels
        })

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset."""
        df['cleaned_message'] = df['message'].apply(preprocess_text)
        return df

    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the phishing detection model."""
        
        logger.info("Creating dataset...")
        self.training_data = self.create_dataset()
        
        logger.info("Preprocessing data...")
        self.training_data = self.preprocess_data(self.training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_data['cleaned_message'], 
            self.training_data['label'], 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.training_data['label']
        )
        
        logger.info("Training TF-IDF vectorizer...")
        # Feature Extraction (TF-IDF)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        logger.info("Training Logistic Regression model...")
        # Train Model
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate Model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        return results

    def save_model(self) -> None:
        """Save the trained model and vectorizer."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be trained before saving")
        
        # Ensure model directory exists
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.vectorizer, VECTORIZER_FILE)
        
        logger.info(f"Model saved to {MODEL_FILE}")
        logger.info(f"Vectorizer saved to {VECTORIZER_FILE}")

def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO)
    
    trainer = PhishGuardTrainer()
    
    # Train the model
    results = trainer.train_model()
    
    # Print results
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save the model
    trainer.save_model()
    print("✅ Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
