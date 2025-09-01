# Synthetic Data?

Created Date: September 1, 2025 7:15 PM
Type of Data: Notes
Language (Type): Python

# 🔹 What is Synthetic Data?

**Synthetic data** is **artificially generated data** that looks and behaves like real data but does not come directly from actual events or users.

For example:

- Instead of collecting **real customer names and addresses**, we can generate **fake but realistic ones**.
- Instead of using **actual sales transactions**, we can **simulate sales records** with products, prices, dates, etc.

It is widely used because:

- It avoids **privacy issues** (no real personal data is exposed).
- It helps when there is **not enough real data** to train models.
- You can **control the distribution** of the data (e.g., more balanced datasets).

---

## 🔹 How to Use Synthetic Data in Python

There are different ways to generate it depending on what you need.

Here are the **main approaches**:

### 1. Using the **Faker** library

```python
from faker import Faker

fake = Faker("en_US")

# Generate fake customer info
print(fake.name())        # John Smith
print(fake.address())     # 123 Main St, Springfield, IL
print(fake.email())       # john.smith@example.com

```

👉 Use when you need **text data** like names, addresses, emails, companies.

---

### 2. Using **NumPy / pandas** for numeric data

```python
import numpy as np
import pandas as pd

# Simulate 1000 sales
data = pd.DataFrame({
    "order_id": np.arange(1, 1001),
    "price": np.random.normal(50, 10, 1000),  # mean=50, std=10
    "quantity": np.random.randint(1, 5, 1000),
})
print(data.head())

```

👉 Use when you need **numeric datasets** like sales, sensor readings, or measurements.

---

### 3. Using **scikit-learn** for machine learning datasets

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

print(X[:5])  # features
print(y[:5])  # labels

```

👉 Use when you need **training datasets** for classification, clustering, regression.

---

### 4. Combining (structured + fake personal info)

```python
customers = pd.DataFrame({
    "CustomerID": range(1, 6),
    "Name": [fake.name() for _ in range(5)],
    "Country": [fake.country() for _ in range(5)],
    "Revenue": np.random.randint(100, 1000, 5)
})
print(customers)

```

---

## 🔹 How You Can Use It

- **Teaching & learning** → practice SQL, Python, ML on safe data.
- **Prototyping apps** → build dashboards, test APIs before real data arrives.
- **Machine learning** → create balanced training data.
- **Testing ETL pipelines** → verify workflows without exposing private info.