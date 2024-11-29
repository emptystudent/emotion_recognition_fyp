import os
import random
from datetime import datetime

class QNA:
    def __init__(self):
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.image_dir = 'static/images'
        self.reset()

    def reset(self):
        self.current_question = 0
        self.score = 0
        self.questions = self.generate_questions()
        self.current_answer = None
        self.emotion_stats = {emotion: {'correct': 0, 'total': 0} for emotion in self.emotions}

    def generate_questions(self):
        questions = []
        for _ in range(10):
            emotion = random.choice(self.emotions)
            emotion_dir = os.path.join(self.image_dir, emotion)
            images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                image = random.choice(images)
                questions.append({
                    'image_path': f'/static/images/{emotion}/{image}',
                    'correct_answer': emotion
                })
        
        return questions

    def get_next_question(self):
        if self.current_question < 10:
            question = self.questions[self.current_question]
            self.current_answer = question['correct_answer']
            return {
                'question_number': self.current_question + 1,
                'image_path': question['image_path'],
                'options': self.emotions
            }
        return None

    def check_answer(self, answer):
        is_correct = answer == self.current_answer
        self.emotion_stats[self.current_answer]['total'] += 1
        if is_correct:
            self.score += 1
            self.emotion_stats[self.current_answer]['correct'] += 1
        
        result = {
            'correct': is_correct,
            'correct_answer': self.current_answer
        }
        
        self.current_question += 1
        return result

    def get_final_score(self):
        return {
            'score': self.score,
            'total': 10
        }

    def get_emotion_stats(self):
        return self.emotion_stats