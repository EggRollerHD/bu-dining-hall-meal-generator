import streamlit as st
import pandas as pd
import pulp
import requests
from bs4 import BeautifulSoup

# --- 1. DATA EXTRACTION (Web Scraping) ---
@st.cache_data # Caches the data so you don't scrape BU's servers on every button click
def scrape_dining_hall(url):
    """
    Template for scraping BU Dining menus. 
    Note: If BU's tables load dynamically via JavaScript, replace requests with Playwright.
    """
    # headers = {'User-Agent': 'Mozilla/5.0'}
    # response = requests.get(url, headers=headers)
    # soup = BeautifulSoup(response.text, 'html.parser')
    
    # Example structure of what your parsed DataFrame will look like:
    mock_data = {
        "Item": ["Scrambled Eggs", "Oatmeal", "Grilled Chicken", "Rice", "Broccoli", "Salmon", "Almond Milk", "Cheese Pizza"],
        "Meal": ["Breakfast", "Breakfast", "Lunch", "Lunch", "Lunch", "Dinner", "Breakfast", "Dinner"],
        "Calories": [140, 150, 165, 205, 50, 200, 60, 285],
        "Protein": [12, 5, 31, 4, 3, 22, 1, 12],
        "Fat": [10, 2.5, 3.5, 0.5, 0, 11, 2.5, 10],
        "Carbs": [1, 27, 0, 45, 10, 0, 8, 36],
        "Ingredients": ["eggs, butter, dairy", "oats, water", "chicken, oil", "white rice", "broccoli", "salmon, fish", "almonds, water", "wheat, dairy, cheese"]
    }
    return pd.DataFrame(mock_data)

# --- 2. THE MATH ENGINE (Macro Optimization) ---
def optimize_meals(df, target_cals, target_p, target_f, target_c, max_servings=2):
    # Initialize the Linear Programming problem
    # We use a minimization problem to minimize the difference between our targets and actuals
    prob = pulp.LpProblem("Macro_Optimizer", pulp.LpMinimize)
    
    # Create variables for each food item (Integer: 0, 1, or 2 servings max)
    food_vars = pulp.LpVariable.dicts("Food", df.index, lowBound=0, upBound=max_servings, cat='Integer')
    
    # Calculate totals based on variable selection
    total_cals = pulp.lpSum([df.loc[i, 'Calories'] * food_vars[i] for i in df.index])
    total_p = pulp.lpSum([df.loc[i, 'Protein'] * food_vars[i] for i in df.index])
    total_f = pulp.lpSum([df.loc[i, 'Fat'] * food_vars[i] for i in df.index])
    total_c = pulp.lpSum([df.loc[i, 'Carbs'] * food_vars[i] for i in df.index])
    
    # Slack variables (Allows the math to find a solution even if it's 5g off target)
    slack_cal = pulp.LpVariable("Slack_Cal", lowBound=0)
    slack_p = pulp.LpVariable("Slack_P", lowBound=0)
    
    # Objective: Minimize deviation from caloric and protein targets
    prob += slack_cal + (slack_p * 10) 
    
    # Constraints (Setting acceptable ranges)
    prob += total_cals - target_cals <= slack_cal
    prob += target_cals - total_cals <= slack_cal
    
    prob += total_p - target_p <= slack_p
    prob += target_p - total_p <= slack_p
    
    # Hard constraints for Fat and Carbs (must be within +/- 10g)
    prob += total_f >= target_f - 10
    prob += total_f <= target_f + 10
    prob += total_c >= target_c - 15
    prob += total_c <= target_c + 15

    # Enforce at least one item from each meal period
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        meal_items = df[df['Meal'] == meal].index
        if len(meal_items) > 0:
            prob += pulp.lpSum([food_vars[i] for i in meal_items]) >= 1

    # Run the solver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Compile results
    results = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i in df.index:
            if food_vars[i].varValue > 0:
                results.append({
                    "Meal": df.loc[i, 'Meal'],
                    "Item": df.loc[i, 'Item'],
                    "Servings": int(food_vars[i].varValue),
                    "Calories": df.loc[i, 'Calories'] * food_vars[i].varValue,
                    "Protein": df.loc[i, 'Protein'] * food_vars[i].varValue
                })
    return pd.DataFrame(results)

# --- 3. USER INTERFACE (Streamlit) ---
st.set_page_config(layout="wide")
st.title("BU Dining Macro Planner 🍽️")

# Sidebar for inputs
with st.sidebar:
    st.header("Daily Targets")
    dining_hall = st.selectbox("Select Dining Hall", [
        "Marciano", "West", "Warren", "Fenway"
    ])
    
    cals = st.number_input("Calories", value=2700)
    protein = st.number_input("Protein (g)", value=150)
    fat = st.number_input("Fat (g)", value=120)
    carbs = st.number_input("Carbs (g)", value=250)
    
    st.header("Dietary Restrictions")
    # Natural language exclusion filter
    exclusions = st.text_input("Exclude ingredients (comma separated)", value="almonds, dairy, fish")
    
url_map = {
    "Marciano": "https://www.bu.edu/dining/location/marciano/#menu",
    "West": "https://www.bu.edu/dining/location/west/#menu",
    "Warren": "https://www.bu.edu/dining/location/warren/#menu",
    "Fenway": "https://bufenway.sodexomyway.com/en-us/locations/the-fenway-dining-hall"
}

# Load and process data
if st.button("Generate Meal Plan"):
    with st.spinner('Scraping menu and running MILP solver...'):
        menu_df = scrape_dining_hall(url_map[dining_hall])
        
        # Filter out unwanted items before the math runs
        if exclusions:
            bad_words = [word.strip().lower() for word in exclusions.split(',')]
            for word in bad_words:
                menu_df = menu_df[~menu_df['Ingredients'].str.contains(word, case=False, na=False)]
                
        # Run optimization
        plan_df = optimize_meals(menu_df, cals, protein, fat, carbs)
        
        if not plan_df.empty:
            st.success("Optimal Meal Plan Found!")
            
            # Display sorted by meal
            for meal in ['Breakfast', 'Lunch', 'Dinner']:
                st.subheader(meal)
                meal_data = plan_df[plan_df['Meal'] == meal]
                st.table(meal_data[['Item', 'Servings', 'Calories', 'Protein']])
                
            st.metric("Total Planned Calories", plan_df['Calories'].sum())
            st.metric("Total Planned Protein", plan_df['Protein'].sum())
        else:
            st.error("Could not find a mathematical combination to hit those exact macros with today's menu. Try adjusting your targets.")
