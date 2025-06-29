"""
Individual task implementations for BitNet v2 evaluation
Implements all tasks mentioned in the paper
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datasets import load_dataset

class BaseTask(ABC):
    """Base class for evaluation tasks"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @abstractmethod
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        """Evaluate the task and return accuracy"""
        pass
    
    def _calculate_completion_probability(self, prompt: str, completion: str) -> float:
        """Calculate the probability of a completion given a prompt"""
        try:
            # Tokenize prompt and completion separately
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
            
            # Combine them
            full_tokens = prompt_tokens + completion_tokens
            if len(full_tokens) > 2048:  # Model's max length
                return float('-inf')
            
            tokens = torch.tensor([full_tokens]).to(self.device)
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(tokens)
                logits = outputs['logits'][0]
            
            # Calculate probability of completion tokens
            log_probs = F.log_softmax(logits, dim=-1)
            
            total_log_prob = 0
            for i, token_id in enumerate(completion_tokens):
                if len(prompt_tokens) + i < len(log_probs):
                    total_log_prob += log_probs[len(prompt_tokens) + i - 1, token_id].item()
            
            return total_log_prob / len(completion_tokens) if completion_tokens else float('-inf')
            
        except Exception as e:
            return float('-inf')
    
    def _calculate_sentence_probability(self, sentence: str) -> float:
        """Calculate the probability of a complete sentence"""
        try:
            tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            if len(tokens) > 2048 or len(tokens) < 2:
                return float('-inf')
            
            tokens = torch.tensor([tokens]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tokens)
                logits = outputs['logits'][0]
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            total_log_prob = 0
            for i in range(1, len(tokens[0])):
                total_log_prob += log_probs[i-1, tokens[0][i]].item()
            
            return total_log_prob / (len(tokens[0]) - 1)
            
        except Exception as e:
            return float('-inf')
    
    def _predict_next_word(self, context: str) -> str:
        """Predict the next word given a context"""
        try:
            tokens = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tokens)
                logits = outputs['logits'][0, -1, :]
            
            # Get the most likely next token
            next_token_id = torch.argmax(logits).item()
            next_word = self.tokenizer.decode([next_token_id])
            
            return next_word.strip()
            
        except Exception as e:
            return ""

class ARCChallengeTask(BaseTask):
    """ARC Challenge task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                correct_answer = item['answerKey']
                
                # Convert answer key to index
                if correct_answer.isdigit():
                    correct_idx = int(correct_answer)
                else:
                    correct_idx = ord(correct_answer.upper()) - ord('A')
                
                if correct_idx >= len(choices):
                    continue
                
                # Calculate probabilities for each choice
                choice_scores = []
                for choice in choices:
                    prompt = f"Question: {question}\nAnswer:"
                    score = self._calculate_completion_probability(prompt, choice)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_idx:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"ARC-C: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in ARC Challenge: {e}")
            return 0.0

class ARCEasyTask(BaseTask):
    """ARC Easy task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                correct_answer = item['answerKey']
                
                # Convert answer key to index
                if correct_answer.isdigit():
                    correct_idx = int(correct_answer)
                else:
                    correct_idx = ord(correct_answer.upper()) - ord('A')
                
                if correct_idx >= len(choices):
                    continue
                
                # Calculate probabilities for each choice
                choice_scores = []
                for choice in choices:
                    prompt = f"Question: {question}\nAnswer:"
                    score = self._calculate_completion_probability(prompt, choice)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_idx:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"ARC-E: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in ARC Easy: {e}")
            return 0.0

class HellaSwagTask(BaseTask):
    """HellaSwag task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("hellaswag", split="validation")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                context = item['ctx']
                choices = item['endings']
                correct_answer = int(item['label'])
                
                # Calculate probabilities for each choice
                choice_scores = []
                for choice in choices:
                    full_text = context + " " + choice
                    score = self._calculate_sentence_probability(full_text)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_answer:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"HellaSwag: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in HellaSwag: {e}")
            return 0.0

class PIQATask(BaseTask):
    """PIQA task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("piqa", split="validation")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                goal = item['goal']
                choices = [item['sol1'], item['sol2']]
                correct_answer = int(item['label'])
                
                # Calculate probabilities for each choice
                choice_scores = []
                for choice in choices:
                    # Format as question-answer
                    prompt = f"Goal: {goal}\nSolution:"
                    score = self._calculate_completion_probability(prompt, choice)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_answer:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"PIQA: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in PIQA: {e}")
            return 0.0

class WinoGrandeTask(BaseTask):
    """WinoGrande task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                sentence = item['sentence']
                choices = [item['option1'], item['option2']]
                correct_answer = int(item['answer']) - 1  # Convert to 0-indexed
                
                # Replace the underscore with each choice
                choice_scores = []
                for choice in choices:
                    completed_sentence = sentence.replace('_', choice)
                    score = self._calculate_sentence_probability(completed_sentence)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_answer:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"WinoGrande: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in WinoGrande: {e}")
            return 0.0

class LAMBADATask(BaseTask):
    """LAMBADA task evaluation"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("lambada", split="test")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                text = item['text']
                
                # Split into context and target word
                words = text.split()
                if len(words) < 2:
                    continue
                
                target_word = words[-1]
                context = ' '.join(words[:-1])
                
                # Generate prediction
                predicted_word = self._predict_next_word(context)
                
                # Check if prediction matches (case-insensitive)
                if predicted_word.lower().strip() == target_word.lower().strip():
                    correct += 1
                total += 1
                
                if total % 100 == 0:
                    print(f"LAMBADA: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in LAMBADA: {e}")
            return 0.0

class BoolQTask(BaseTask):
    """BoolQ task evaluation (additional task)"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("boolq", split="validation")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                passage = item['passage']
                question = item['question']
                answer = item['answer']  # True or False
                
                # Format prompt
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                
                # Calculate probabilities for Yes/No
                yes_score = self._calculate_completion_probability(prompt, " Yes")
                no_score = self._calculate_completion_probability(prompt, " No")
                
                # Predict based on higher probability
                predicted_yes = yes_score > no_score
                
                if predicted_yes == answer:
                    correct += 1
                total += 1
                
                if total % 100 == 0:
                    print(f"BoolQ: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in BoolQ: {e}")
            return 0.0

class OpenBookQATask(BaseTask):
    """OpenBookQA task evaluation (additional task)"""
    
    def evaluate(self, num_samples: Optional[int] = None) -> float:
        try:
            dataset = load_dataset("openbookqa", split="test")
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for item in dataset:
                question = item['question_stem']
                choices = [choice['text'] for choice in item['choices']]
                correct_answer = item['answerKey']
                
                # Convert answer key to index
                correct_idx = ord(correct_answer.upper()) - ord('A')
                
                if correct_idx >= len(choices):
                    continue
                
                # Calculate probabilities for each choice
                choice_scores = []
                for choice in choices:
                    prompt = f"Question: {question}\nAnswer:"
                    score = self._calculate_completion_probability(prompt, choice)
                    choice_scores.append(score)
                
                # Predict the choice with highest score
                if choice_scores:
                    predicted = choice_scores.index(max(choice_scores))
                    if predicted == correct_idx:
                        correct += 1
                    total += 1
                
                if total % 100 == 0:
                    print(f"OpenBookQA: Processed {total} samples, accuracy: {correct/total:.3f}")
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in OpenBookQA: {e}")
            return 0.0
