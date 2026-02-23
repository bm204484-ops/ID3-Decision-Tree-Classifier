import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Decision Tree", layout="centered")
st.title("ID3 Decision Tree Classifier")

data = pd.DataFrame({
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "High", "Normal", "High", "Normal", "High"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
})

st.subheader("Training Dataset")
st.dataframe(data, use_container_width=True)

def entropy(col):
    counts = np.unique(col, return_counts=True)[1]
    ent = 0
    for c in counts:
        p = c / len(col)
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum((len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target]) for v in vals)
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    
    best = max(attrs, key=lambda x: info_gain(df, x, target))
    tree = {best: {}}
    for v in df[best].unique():
        subset = df[df[best] == v]
        if subset.empty:
            tree[best][v] = df[target].mode()[0]
        else:
            tree[best][v] = id3(subset, target, [a for a in attrs if a != best])
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    key = next(iter(tree))
    val = sample.get(key)
    if val not in tree[key]:
        return "Unknown"
    return predict(tree[key][val], sample)

if st.button("Generate Decision Tree"):
    tree_result = id3(data, "PlayTennis", ["Outlook", "Humidity"])
    st.session_state.tree = tree_result
    st.subheader("Generated Decision Tree Structure")
    st.json(tree_result)

if "tree" in st.session_state:
    st.divider()
    st.subheader("Make a Prediction")
    col1, col2 = st.columns(2)
    with col1:
        o = st.selectbox("Outlook", data["Outlook"].unique())
    with col2:
        h = st.selectbox("Humidity", data["Humidity"].unique())
    
    if st.button("Predict"):
        result = predict(st.session_state.tree, {"Outlook": o, "Humidity": h})
        st.write(f"**Outlook:** {o} | **Humidity:** {h}")
        if result == "Yes":
            st.success(f"Result: {result} (Go play!)")
        else:
            st.error(f"Result: {result} (Stay
