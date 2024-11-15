import pandas as pd
import random

# Lists of example toxic and non-toxic comments
toxic_comments = [
    "You are a complete idiot", "I hate everything you do", "This is the worst thing ever",
    "You're so stupid", "Nobody likes you", "You are a failure", "Get lost, nobody wants you here",
    "You're a disgrace", "Stop being such a loser", "I wish you were gone", "This is trash"
]

non_toxic_comments = [
    "Great job!", "This is fantastic", "I really appreciate your effort",
    "Wonderful experience", "Keep up the good work", "Thank you so much",
    "That was very helpful", "I enjoyed this", "You did amazing", "Great guide, thanks!",
    "Excellent work, well done!"
]

# Create a DataFrame
data = []
for _ in range(5000):
    # 5000 toxic comments
    data.append([random.choice(toxic_comments), 1])
    # 5000 non-toxic comments
    data.append([random.choice(non_toxic_comments), 0])

# Shuffle the dataset
random.shuffle(data)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['comment', 'label'])

# Save to CSV
df.to_csv('comments.csv', index=False)
print("Sample dataset created as 'comments.csv'")
