import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


DATA_PATH = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv"

def train_decision_stage_model(df_train):
    if 'text' not in df_train.columns:
        df_train = df_train.copy()
        df_train['text'] = df_train.get('title', '').fillna('') + ' ' + df_train.get('summary', '').fillna('')
    else:
        df_train = df_train.copy()

    if df_train.empty:
        print('No labeled data available for training.')
        return None, None

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_known = vectorizer.fit_transform(df_train['text'])
    y_known = df_train['outcome_binary']

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known, test_size=0.2, random_state=42, stratify=y_known
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known, test_size=0.2, random_state=42
        )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print('\nClassification report for decision_stage prediction:')
    print(classification_report(y_test, preds))

    return model, vectorizer

def predict_outcome(model, vectorizer, df_predict):
    if len(df_predict) == 0:
        print('No cases in Investigation stage to predict.')
    else:
        X_predict = vectorizer.transform(df_predict['text'])
        predict_outcomes = model.predict(X_predict)
        predict_proba = model.predict_proba(X_predict)

        # model.classes_ is alphabetical: ['Approved', 'Rejected']
        # so column 0 = Approved, column 1 = Rejected
        approved_idx = list(model.classes_).index('Approved')
        rejected_idx = list(model.classes_).index('Rejected')

        df_predict = df_predict.copy()
        df_predict['predicted_outcome']  = predict_outcomes
        df_predict['prob_approved']      = predict_proba[:, approved_idx]
        df_predict['prob_rejected']      = predict_proba[:, rejected_idx]
        df_predict['confidence']         = np.where(df_predict['prob_approved'] >= 0.5, df_predict['prob_approved'], 1 - df_predict['prob_approved'])
        print(f"\nPredicted outcomes for {len(df_predict):,} investigation cases:")
        print(df_predict['predicted_outcome'].value_counts())
        print()

        print("TOP 15 CASES MOST LIKELY TO BE REJECTED:")
        print("(These are active investigations the model thinks will end badly)")
        cols = ['article_id', 'year', 'sector', 'case_type', 
            'title', 'prob_rejected']
        top_rejected = (df_predict
                    .sort_values('prob_rejected', ascending=False)
                    .head(15)[cols])
        print(top_rejected.to_string(index=False))

        print("\nTOP 15 CASES MOST LIKELY TO BE APPROVED:")
        top_approved = (df_predict
                    .sort_values('prob_approved', ascending=False)
                    .head(15)[cols[:-1] + ['prob_approved']])
        print(top_approved.to_string(index=False))
        return df_predict


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        df_train = df[df['outcome_binary'].isin(['Approved', 'Rejected'])].copy()
        df_predict = df[df['decision_stage'] == 'Investigation'].copy()
        model, vec = train_decision_stage_model(df_train)
        if model and vec:
            new_predict_cases = predict_outcome(model, vec, df_predict)

            # Map predicted outcomes back to the main table to complete outcome_binary
            df_complete = df.copy()
            df_complete = df_complete.merge(
                new_predict_cases[['article_id', 'predicted_outcome']],
                on='article_id',
                how='left'
            )
            # Fill missing outcome_binary with predicted_outcome for Investigation cases
            mask = df_complete['outcome_binary'].isna() & df_complete['predicted_outcome'].notna()
            df_complete.loc[mask, 'outcome_binary'] = df_complete.loc[mask, 'predicted_outcome']
            df_complete.drop(columns=['predicted_outcome'], inplace=True)

            print(f"\noutcome_binary fill rate after prediction: "
                  f"{df_complete['outcome_binary'].notna().sum():,}/{len(df_complete):,} "
                  f"({df_complete['outcome_binary'].notna().mean()*100:.1f}%)")
            print(df_complete['outcome_binary'].value_counts())

            # Save predictions-only file
            new_predict_cases.to_csv(os.path.join(os.path.dirname(DATA_PATH), 'predicted_investigation_outcomes.csv'), index=False)

            # Save complete table with outcome_binary fully populated
            complete_path = os.path.join(os.path.dirname(DATA_PATH), 'final_competition_cases.csv')
            df_complete.to_csv(complete_path, index=False)
            print(f"\nComplete table saved → {complete_path}")
