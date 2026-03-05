import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INPUT_CSV = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data.csv"
OUTPUT_CSV = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data_classified.csv"
OUTPUT_CSV_COMPETITION = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data_competition_cases.csv"
# Competition case types
COMPETITION_TYPES = ['merger', 'antitrust', 'state aid']

# Normalize variant spellings → canonical forms
CASE_TYPE_ALIASES = {
    'mergers': 'merger',
    'merger regulation': 'merger',
    'state aids': 'state aid',
}

# Class for everything that is NOT competition
NOT_COMPETITION = 'not_competition'


def main():
    df = pd.read_csv(INPUT_CSV)

    # Normalize case_type variants
    df['case_type'] = df['case_type'].str.strip().str.lower()
    df['case_type'] = df['case_type'].replace(CASE_TYPE_ALIASES)

    # Combine title + summary as text feature
    df['text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')

    # ═══════════════════════════════════════════════════════════════════════
    # Build labels: 4 classes (merger / state aid / antitrust / not_competition)
    # ═══════════════════════════════════════════════════════════════════════

    has_case = df['case_type'].notna()
    labeled = df[has_case].copy()
    unlabeled = df[~has_case].copy()

    # Map: competition types stay as-is, everything else → not_competition
    labeled['label'] = labeled['case_type'].apply(
        lambda x: x if x in COMPETITION_TYPES else NOT_COMPETITION
    )

    print(f"Total rows: {len(df)}")
    print(f"Labeled rows: {len(labeled)}")
    print(f"Unlabeled rows: {len(unlabeled)}")
    print(f"\nLabel distribution (training data):")
    print(labeled['label'].value_counts().to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # Train 4-class classifier
    # ═══════════════════════════════════════════════════════════════════════

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_labeled = tfidf.fit_transform(labeled['text'])
    y_labeled = labeled['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Classification Report (on labeled test set) ---")
    print(classification_report(y_test, y_pred))

    # ═══════════════════════════════════════════════════════════════════════
    # Predict unlabeled rows
    # ═══════════════════════════════════════════════════════════════════════

    if len(unlabeled) > 0:
        X_unlabeled = tfidf.transform(unlabeled['text'])
        predicted_labels = clf.predict(X_unlabeled)
        predicted_probs = clf.predict_proba(X_unlabeled)
        confidence = predicted_probs.max(axis=1)

        unlabeled['label'] = predicted_labels
        unlabeled['case_type_confidence'] = confidence

        print(f"\nPredictions for {len(unlabeled)} unlabeled rows:")
        print(pd.Series(predicted_labels).value_counts().to_string())
        print(f"Average confidence: {confidence.mean():.3f}")
        print(f"Low confidence (<0.5): {(confidence < 0.5).sum()} cases")

    # ═══════════════════════════════════════════════════════════════════════
    # Merge results back
    # ═══════════════════════════════════════════════════════════════════════

    # For labeled rows: assign label and set confidence=1.0 (ground truth)
    df.loc[labeled.index, 'label'] = labeled['label']
    df.loc[labeled.index, 'case_type_confidence'] = 1.0

    # For unlabeled rows: assign predicted label and confidence
    if len(unlabeled) > 0:
        df.loc[unlabeled.index, 'label'] = unlabeled['label']
        df.loc[unlabeled.index, 'case_type_confidence'] = unlabeled['case_type_confidence']

    # Derive is_competition and case_type from label
    df['is_competition'] = df['label'].isin(COMPETITION_TYPES)

    # For competition cases: set case_type to the predicted/known type
    # For non-competition: keep original case_type (if any) or leave as-is
    comp_mask = df['is_competition']
    df.loc[comp_mask, 'case_type'] = df.loc[comp_mask, 'label']

    df.drop(columns=['text', 'label'], inplace=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Save & Summary
    # ═══════════════════════════════════════════════════════════════════════

   

    df.to_csv(OUTPUT_CSV, index=False)

    df_competion= df[
    (df["case_type"] == "merger") |
    (df["case_type"] == "state aid") |
    (df["case_type"] == "antitrust")
    ].copy()


    print(f"\nCompetition cases extracted: {len(df_competion)}")
    print("saving competition cases to separate file...")
    df_competion.to_csv(OUTPUT_CSV_COMPETITION, index=False)

    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total rows:              {len(df)}")
    print(f"  is_competition=True:     {df['is_competition'].sum()}")
    print(f"  is_competition=False:    {(~df['is_competition']).sum()}")
    print(f"\nCase type breakdown (competition cases):")
    print(df[df['is_competition']]['case_type'].value_counts().to_string())
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
