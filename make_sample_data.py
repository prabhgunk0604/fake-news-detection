"""
Ye script sirf DEMO/TESTING ke liye chhota sample dataset banata hai.
Real project ke liye Kaggle se "Fake and Real News Dataset" (True.csv + Fake.csv)
download karke isi folder mein daal do -- code automatically wahi use kar lega
agar True.csv aur Fake.csv files present hain.

Kaggle link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""
import pandas as pd
import random

real_templates = [
    "The government announced a new policy on {topic} today, according to officials.",
    "Scientists at {org} published a study on {topic} in a peer-reviewed journal.",
    "The stock market showed {topic} trends this week as reported by financial analysts.",
    "Local authorities confirmed that {topic} infrastructure projects will begin next month.",
    "According to the ministry of health, {topic} vaccination rates have improved this year.",
    "The parliament passed a bill regarding {topic} after months of debate.",
    "Researchers at {org} have found new evidence related to {topic} in a controlled experiment.",
    "The central bank released its quarterly report discussing {topic} and inflation trends.",
]

fake_templates = [
    "SHOCKING: You won't believe what {topic} secretly does to your body!!!",
    "BREAKING: Aliens confirmed to be behind {topic}, government hides truth!",
    "Doctors HATE this one trick about {topic} that big companies don't want you to know.",
    "Forwarded message: {topic} will destroy the economy tomorrow, share before it's deleted!",
    "Miracle cure for {topic} discovered by a man in his garage, doctors are furious.",
    "Celebrity secretly confesses {topic} conspiracy in leaked video, media refuses to show it.",
    "URGENT!!! {topic} causes instant death, mainstream media hiding the truth from you.",
    "Anonymous insider reveals {topic} is actually a hoax created by world leaders.",
]

topics = ["climate change", "the economy", "vaccines", "5G towers", "elections", "AI technology",
          "the education system", "cryptocurrency", "the housing market", "space exploration",
          "renewable energy", "public health", "immigration policy", "the stock market"]
orgs = ["MIT", "Oxford University", "a national research lab", "Stanford University", "a leading university"]

def generate(n, templates, label):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        text = t.format(topic=random.choice(topics), org=random.choice(orgs))
        rows.append({"text": text, "label": label})
    return rows

random.seed(42)
data = generate(300, real_templates, "REAL") + generate(300, fake_templates, "FAKE")
random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv("news_dataset.csv", index=False)
print("Sample dataset ban gaya: news_dataset.csv, total rows:", len(df))
print(df['label'].value_counts())
