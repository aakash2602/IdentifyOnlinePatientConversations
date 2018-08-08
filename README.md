# IdentifyOnlinePatientConversations

Problem Statement:
Conversation log of people asking question related to some disease
We need to classify the documents into 2 types - conversation related to cancer and all the other conversations.

2 Approachs:
1. BOW -> LSI -> Logistic Regression
2. BOW -> TFIDF -> Shallow NN

Results:
Approach 2 works best with 90%+ accuracy on Test Data
ROC curve is used for evaluation
