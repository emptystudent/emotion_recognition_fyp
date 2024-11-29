import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import os

class Dashboard:
    def __init__(self):
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.load_data()

    def load_data(self):
        try:
            with open('user_responses.json', 'r') as file:
                self.raw_data = json.load(file)
            records = []
            for emotion, stats in self.raw_data.items():
                for timestamp, result in stats.get('history', {}).items():
                    records.append({
                        'emotion': emotion,
                        'date': datetime.fromtimestamp(float(timestamp)),
                        'correct': result['correct'],
                        'predicted': result['predicted']
                    })
            self.df = pd.DataFrame(records)
        except FileNotFoundError:
            self.raw_data = {}
            self.df = pd.DataFrame()

    def generate_emotion_chart(self, start_date=None, end_date=None, selected_emotions=None):
        if self.df.empty:
            return None

        filtered_df = self.df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        if selected_emotions:
            filtered_df = filtered_df[filtered_df['emotion'].isin(selected_emotions)]

        # Calculate accuracy for each emotion
        accuracy_data = {}
        for emotion in self.emotions:
            emotion_data = filtered_df[filtered_df['emotion'] == emotion]
            total = len(emotion_data)
            correct = len(emotion_data[emotion_data['correct'] == True])
            accuracy = (correct / total * 100) if total > 0 else 0
            accuracy_data[emotion] = accuracy

        # Create bar chart
        plt.figure(figsize=(12, 6))
        emotions = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())
        
        bars = plt.bar(emotions, accuracies, color='lightblue', edgecolor='blue')
        
        plt.xlabel('Emotions')
        plt.ylabel('Accuracy (%)')
        plt.title('Average Accuracy (%)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Ensure the static/images directory exists
        os.makedirs('static/images', exist_ok=True)
        
        # Save the chart
        plt.savefig('static/images/emotion_accuracy.png')
        plt.close()

        return accuracy_data

    def get_test_history(self, start_date=None, end_date=None):
        if self.df.empty:
            return []
        
        filtered_df = self.df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= end_date]

        history = []
        for timestamp in sorted(filtered_df['date'].unique()):
            session_data = filtered_df[filtered_df['date'] == timestamp]
            
            emotion_accuracies = {}
            overall_correct = 0
            overall_total = 0
            
            for emotion in self.emotions:
                emotion_data = session_data[session_data['emotion'] == emotion]
                total = len(emotion_data)
                correct = len(emotion_data[emotion_data['correct'] == True])
                accuracy = (correct / total * 100) if total > 0 else 0
                emotion_accuracies[emotion] = accuracy
                overall_correct += correct
                overall_total += total

            overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
            
            history.append({
                'timestamp': timestamp,
                'overall_accuracy': overall_accuracy,
                **emotion_accuracies
            })

        return history

    def delete_entry(self, timestamp):
        try:
            timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            self.df = self.df[self.df['date'] != timestamp_dt]
            # Update raw_data and save to file
            # Add implementation here if needed
            return True
        except Exception as e:
            print(f"Error deleting entry: {e}")
            return False

