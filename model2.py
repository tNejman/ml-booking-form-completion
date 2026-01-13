import re
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

def extract_amenities_from_description(description: str):
    regex_map = {}
    with open("amenity_patterns.json") as file:
        regex_map = json.load(file)
    if len(regex_map) == 0:
        raise Exception("Loaded 0 regex patterns.")
    
    compiled_patterns = {key: re.compile(pattern) for key, pattern in regex_map.items()}
    found_amenities = set()
    
    for category, pattern in compiled_patterns.items():
        if category not in found_amenities:
            if pattern.search(description):
                found_amenities.add(category)
            
    return sorted(list(found_amenities))

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        word_to_num = {
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10", "studio": "0",
        }

        for word, num in word_to_num.items():
            text = re.sub(r"\b" + word + r"\b", num, text)

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class PredictionModel:
    TARGETS_CLASS = ["room_type", "property_type", "bathrooms_text"]
    TARGETS_REG = ["bedrooms", "beds", "accommodates"]
    
    def learn(self, df_train: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement learn method.")

    def predict(self, description: str) -> dict[str, str]:
        raise NotImplementedError("Subclasses must implement predict method.")


class BasePredictionModel(PredictionModel):
    def __init__(self):
        self.stats = {}

    def learn(self, df_train: pd.DataFrame):
        print(f"  [BaseModel] Learning naive statistics from {len(df_train)} rows...")
        
        for target in self.TARGETS_CLASS:
            if target in df_train.columns:
                valid = df_train[target].dropna()
                if not valid.empty:
                    self.stats[target] = valid.mode()[0]
                else:
                    self.stats[target] = "Unknown"

        for target in self.TARGETS_REG:
            if target in df_train.columns:
                valid = df_train[target].dropna()
                if not valid.empty:
                    self.stats[target] = int(valid.median())
                else:
                    self.stats[target] = 0
        return self

    def predict(self, description: str) -> dict[str, str]:
        self.stats["amenities"] = extract_amenities_from_description(description)
        self.stats["model_version"] = "baseline"
        return self.stats


class AdvancedPredictionModel(PredictionModel):
    def __init__(self):
        self.pipelines = {}
        self.tfidf_params = {
            "max_features": 5000,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "token_pattern": r"(?u)\b\w+\b",
            "min_df": 3,
        }

    def learn(self, df_train: pd.DataFrame):
        print(f"  [AdvancedModel] Training ML pipelines on {len(df_train)} rows...")
        
        for target in self.TARGETS_CLASS:
            self._train_pipeline(df_train, target, model_type='classification')
        for target in self.TARGETS_REG:
            self._train_pipeline(df_train, target, model_type='regression')
        
        return self

    def _train_pipeline(self, df, target, model_type):
        data = df.dropna(subset=[target, "description"])
        if data.empty:
            print(f"    Warning: No data for {target}")
            return

        X = data["description"]
        y = data[target]

        if model_type == 'classification':
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        else:
            clf = Ridge(alpha=0.5)

        pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(**self.tfidf_params)),
            ("model", clf),
        ])

        pipeline.fit(X, y)
        self.pipelines[target] = pipeline

    def predict(self, description: str) -> dict[str, str]:
        predictions = {}
        input_data = pd.Series([description])

        for target, pipeline in self.pipelines.items():
            try:
                pred = pipeline.predict(input_data)[0]
                if target in self.TARGETS_REG:
                    pred = int(round(pred))
                
                predictions[target] = pred
            except Exception:
                predictions[target] = None
        
        predictions["amenities"] = extract_amenities_from_description(description)
        predictions["model_version"] = "advanced"
        return predictions

def evaluate_models(base_model, adv_model, df_test):
    print("\n--- Evaluation on Test Set ---")
    
    classification_targets = ['room_type', 'property_type', 'bathrooms_text']
    regression_targets = ['bedrooms', 'beds', 'accommodates']
    results = []
    for target in classification_targets + regression_targets:
        if not target in df_test.columns:
            raise Exception("Illegal column:" + target)
        
        valid_test = df_test.dropna(subset=[target, "description"])
        if not valid_test.empty:
            y_true = valid_test[target]
            y_pred_base = [base_model.predict("") [target]] * len(valid_test)            
            y_pred_adv = valid_test["description"].apply(lambda x: adv_model.predict(x).get(target, 0))

            if target in regression_targets:
                metric = mean_absolute_error
                metric_name = "MAE"
            else:
                metric = accuracy_score
                metric_name = "Accuracy"
                
            score_base = metric(y_true, y_pred_base)
            score_adv = metric(y_true, y_pred_adv)
            
            if target in regression_targets:
                improvement = score_base - score_adv
            else:
                improvement = score_adv - score_base
                
            results.append({
                "Target": target,
                "Metric": metric_name,
                "Base model score": round(score_base, 4),
                "Advanced model score": round(score_adv, 4),
                "Improvement": round(improvement, 4),
                "Status": "SUKCES" if improvement > 0 else "PORAŻKA"
            })
    print(pd.DataFrame(results))
                

def train_and_evaluate(base_model, advanced_model, csv_path="listings1.csv", train_ratio=0.8, save_path="models.pkl"):
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    print(f"1. Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    df.dropna(subset=["description"])

    print(f"2. Splitting data (Train: {train_ratio:.0%}, Test: {1-train_ratio:.0%})...")
    df_train, df_test = train_test_split(df, train_size=train_ratio, random_state=42)

    base_model.learn(df_train)
    advanced_model.learn(df_train)

    artifacts = {
        "base_model": base_model, 
        "advanced_model": advanced_model
    }
    joblib.dump(artifacts, save_path)
    print(f"\nModels saved to {save_path}")

    evaluate_models(base_model, advanced_model, df_test)