import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate_roi(row):
    return (row['Conversions'] * 100) / row['Spend ($)']

def make_budget_decisions(df):
    df['ROI'] = df.apply(calculate_roi, axis=1)
    avg_roi = df['ROI'].mean()

    decisions = []
    for _, row in df.iterrows():
        if row['ROI'] < 0.75 * avg_roi:
            action = "decrease"
        elif row['ROI'] > 1.25 * avg_roi:
            action = "increase"
        else:
            action = "maintain"

        decisions.append({
            "Platform": row['Platform'],
            "Device Type": row['Device Type'],
            "Audience Type": row['Audience Type'],
            "Spend ($)": row['Spend ($)'],
            "ROI": round(row['ROI'], 2),
            "Action": action
        })
    return pd.DataFrame(decisions)

def generate_explanations(df_decisions):
    explanations = []
    for _, row in df_decisions.iterrows():
        prompt = f'''
        You're an AI ad strategist. A campaign on {row['Platform']} targeting {row['Audience Type']} users on {row['Device Type']} has an ROI of {row['ROI']}.
        You chose to {row['Action']} the budget. Briefly explain why.
        '''
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            explanation = f"Error: {e}"
        explanations.append(explanation)

    df_decisions["Explanation"] = explanations
    return df_decisions