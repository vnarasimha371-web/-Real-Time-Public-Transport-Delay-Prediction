import pandas as pd
import matplotlib.pyplot as plt

# 1. Create a dictionary with the results we found
data = {
    "Model": ["Random Forest", "ARIMA", "LSTM"],
    "MAE": [2.46, 2.39, 2.88]
}

# 2. Convert to a DataFrame
df = pd.DataFrame(data)

# 3. Plot the results as a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(df["Model"], df["MAE"], color=["blue", "green", "red"])

# 4. Add titles and labels
plt.title("Model Comparison - Mean Absolute Error (MAE)")
plt.xlabel("Model")
plt.ylabel("MAE (minutes)")
plt.ylim(0, 4) # Set y-axis limit so the bars look nice

# 5. Add the exact numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()

# 6. Save the graph
plt.savefig("model_comparison.png", dpi=150)
plt.show()

print("Success! model_comparison.png created.")