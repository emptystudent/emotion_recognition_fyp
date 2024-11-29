from flask import Flask, render_template, request, jsonify, Response
from QNA import QNA
from Emotion_Recognition import EmotionRecognition
import json
import os
from datetime import datetime
import cv2
import numpy as np
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
qna = QNA()
emotion_recognition = EmotionRecognition()

# Update base URL - remove /api suffix
JAMAI_BASE_URL = "https://api.jamaibase.com"  
JAMAI_API_KEY = "jamai_sk_09081aedfccde72a8cfa4bc4db0ff23fa5bd47406c885b77"
PROJECT_ID = "proj_2e6b08f124289d82c1e430d3"
AGENT_ID = "shunxin"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qna')
def qna_page():
    qna.reset()
    return render_template('qna.html')

@app.route('/get_question')
def get_question():
    question = qna.get_next_question()
    if question:
        return jsonify(question)
    return jsonify({'complete': True})

@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.get_json()
    result = qna.check_answer(data['answer'])
    return jsonify(result)

@app.route('/get_score')
def get_score():
    return jsonify(qna.get_final_score())

@app.route('/save_quiz_results', methods=['POST'])
def save_quiz_results():
    stats = qna.get_emotion_stats()
    score = qna.get_final_score()
    
    test_data = {
        'test_taken': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'emotion_stats': stats,
        'overall_accuracy': (score['score'] / score['total']) * 100
    }
    
    history = load_history()
    history.append(test_data)
    save_history(history)
    return jsonify({'success': True})

@app.route('/dashboard')
def dashboard():
    history = load_history()
    # Transform the history data to include emotion_accuracies
    formatted_history = []
    for test in history:
        emotion_accuracies = {}
        for emotion in test['emotion_stats']:
            stats = test['emotion_stats'][emotion]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
            else:
                accuracy = 0
            emotion_accuracies[emotion] = accuracy

        formatted_test = {
            'test_taken': test['test_taken'],
            'emotion_accuracies': emotion_accuracies,
            'emotion_stats': test['emotion_stats'],
            'overall_accuracy': test['overall_accuracy'],
            'correct_count': sum(stats['correct'] for stats in test['emotion_stats'].values())
        }
        formatted_history.append(formatted_test)

    return render_template('dashboard.html', stats={'history': formatted_history})

@app.route('/delete_test/<int:index>', methods=['POST'])
def delete_test(index):
    try:
        history = load_history()
        if 0 <= index < len(history):
            history.pop(index)
            save_history(history)
            return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/emotion_recognition')
def emotion_recognition_page():
    return render_template('emotion_recognition.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        emotion_recognition.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        # Read and convert image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze image
        results = emotion_recognition.analyze_image(image)
        return jsonify({'emotions': results})
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/chat', methods=['POST'])
def chat():
    print("Chat endpoint hit")
    try:
        data = request.get_json()
        user_message = data.get('message')
        print(f"Received message: {user_message}")
        
        if not user_message:
            print("Error: Empty message received")
            return jsonify({'error': 'Empty message'}), 400

        # JamAI API call
        headers = {
            'Authorization': f'Bearer {JAMAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Updated payload structure
        payload = {
            'messages': [{
                'role': 'user',
                'content': user_message
            }],
            'projectId': PROJECT_ID,
            'agentId': AGENT_ID,
            'stream': False,
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        print("Making JamAI API call...")
        # Try different endpoint format
        api_url = f"{JAMAI_BASE_URL}/chat"  # Simplified endpoint
        print(f"API URL: {api_url}")  # Debug print
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            assistant_response = result.get('response', '')
            print(f"Response: {assistant_response}")
            return jsonify({'response': assistant_response})
        else:
            error_message = f"JamAI API Error: {response.text}"
            print(error_message)
            return jsonify({'error': error_message}), response.status_code
            
    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_history():
    if os.path.exists('static/test_history.json'):
        with open('static/test_history.json', 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open('static/test_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
