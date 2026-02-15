# ------------------------------------------------------
# SPORTS vs POLITICS CLASSIFIER
# With Accuracy + Confusion Matrix
# ------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------- SAMPLE DATASET ---------------- #

sports_docs = [
    "India won the cricket match",
    "The football team scored three goals",
    "Olympic games start next month",
    "He hit a brilliant century",
    "The tennis player won the championship",
    "The coach praised the players",
    "Basketball finals were exciting",
    "The athlete broke the world record",
    "The team celebrated the victory",
    "Hockey tournament begins tomorrow"
]

politics_docs = [
    "The election results were announced",
    "The parliament passed a new bill",
    "Government launched a new policy",
    "The president gave a speech",
    "Political parties started campaigning",
    "The minister held a press conference",
    "The budget was presented today",
    "The leader addressed the nation",
    "New law was implemented",
    "The senate approved the proposal"
]

docs = sports_docs + politics_docs
labels = ["SPORTS"] * len(sports_docs) + ["POLITICS"] * len(politics_docs)


# ---------------- TF-IDF FEATURES ---------------- #

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)


# ---------------- MODELS ---------------- #

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}


print("\nModel Performance:\n")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=["SPORTS", "POLITICS"])

    print(f"===== {name} =====")
    print(f"Accuracy: {round(acc*100,2)} %\n")

    print("Confusion Matrix:")
    print("               Predicted")
    print("             SPORTS  POLITICS")
    print(f"Actual SPORTS     {cm[0][0]}        {cm[0][1]}")
    print(f"Actual POLITICS   {cm[1][0]}        {cm[1][1]}")
    print("\n")


# ---------------- INTERACTIVE TEST ---------------- #

print("Enter custom sentence (type exit to stop)\n")

while True:

    text = input("Input: ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])

    prediction = models["SVM"].predict(vec)[0]

    print("Predicted Class:", prediction, "\n")
