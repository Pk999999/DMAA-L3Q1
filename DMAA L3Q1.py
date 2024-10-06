import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

students_df = pd.read_csv('student_attendance_performance.csv')
X = students_df[['Attendance', 'Performance']]
y = students_df['Likely_Dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

students_df['Predicted_Dropout'] = knn.predict(X)
print("\nStudents likely to drop out or fail early:")
dropout_students = students_df[students_df['Predicted_Dropout'] == True]
print(dropout_students[['Student_ID', 'Attendance', 'Performance', 'Subject']])

# Display students likely to pass
print("\nStudents likely to pass:")
passing_students = students_df[students_df['Predicted_Dropout'] == False]
print(passing_students[['Student_ID', 'Attendance', 'Performance', 'Subject']])