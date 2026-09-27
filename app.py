import streamlit as st
import pandas as pd
import pulp
import requests
from bs4 import BeautifulSoup

# --- 1. DATA EXTRACTION (Real Web Scraper) ---
@st.cache_data(ttl=3600) # Caches the live data for 1 hour so it doesn't overload BU servers
def scrape_dining_hall(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        parsed_data = []
        current_meal = "Lunch" # Default fallback
        
        # Parse standard HTML structures for college dining menus
        for element in soup.find_all(['h3', 'h4', 'li', 'div'], class_=['meal-period', 'menu-item', 'menu-details', 'item-name']):
            text = element.get_text(strip=True).lower()
            
            # Track which meal period we are currently parsing
            if 'breakfast' in text: current_meal = 'Breakfast'
            elif 'lunch' in text: current_meal = 'Lunch'
            elif 'dinner' in text: current_meal = 'Dinner'
            
            # Extract item details
            if 'menu-item' in element.get('class', []) or 'menu-details' in element.get('class', []):
                try:
                    name_elem = element.find(class_='item-name') or element.find('a')
                    name = name_elem.get_text(strip=True) if name_elem else "Unknown Item"
                    
                    # Extract Macros (stripping out text like 'g' or 'kcal')
                    cals = element.find(class_='calories')
                    cals = int(''.join(filter(str.isdigit, cals.get_text()))) if cals else 0
                    
                    protein = element.find(class_='protein')
                    protein = int(''.join(filter(str.isdigit, protein.get_text()))) if protein else 0
                    
                    fat = element.find(class_='fat') or element.find(class_='total-fat')
                    fat = int(''.join(filter(str.isdigit, fat.get_text()))) if fat else 0
                    
                    carbs = element.find(class_='carbohydrates') or element.find(class_='total-carbs')
                    carbs = int(''.join(filter(str.isdigit, carbs.get_text()))) if carbs else 0
                    
                    ingredients_elem = element.find(class_='ingredients')
                    ingredients = ingredients_elem.get_text(strip=True).lower() if ingredients_elem else ""
                    
                    if name != "Unknown Item" and cals > 0:
                        parsed_data.append({
                            "Item": name,
                            "Meal": current_meal,
                            "Calories": cals,
                            "Protein": protein,
                            "Fat": fat,
                            "Carbs": carbs,
                            "Ingredients": ingredients
                        })
                except Exception:
                    continue
                    
        df = pd.DataFrame(parsed_data)
        
        # Fallback if the live site uses JavaScript blocking or structure changes
        if df.empty:
            st.warning("Live scraping returned 0 items (HTML structure may have changed). Using fallback data to prevent crash.")
            mock_data = {
                "Item": ["Scrambled Eggs", "Oatmeal", "Grilled Chicken", "Rice", "Broccoli", "Salmon", "Almond Milk", "Cheese Pizza"],
                "Meal": ["Breakfast", "Breakfast", "Lunch", "Lunch", "Lunch", "Dinner", "Breakfast", "Dinner"],
                "Calories": [140, 150, 165, 205, 50, 200, 60, 285],
                "Protein": [12, 5, 31, 4, 3, 22, 1, 12],
                "Fat": [10, 2.5, 3.5, 0.5, 0, 11, 2.5, 10],
                "Carbs": [1, 27, 0, 45, 10, 0, 8, 36],
                "Ingredients": ["eggs, butter, dairy", "oats, water", "chicken, oil", "white rice", "broccoli", "salmon, fish", "almonds, water", "wheat, dairy, cheese"]
            }
            df = pd.DataFrame(mock_data)
            
        return df

    except Exception as e:
        st.error(f"Failed to fetch menu: {e}")
        return pd.DataFrame()

# --- 2. THE MATH ENGINE (Macro Optimization) ---
def optimize_meals(df, target_cals, target_p, target_f, target_c, max_servings=2):
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.reset_index(drop=True)
    valid_indices = df.index.tolist()
    
    prob = pulp.LpProblem("Macro_Optimizer", pulp.LpMinimize)
    
    # Restored to 3.x compatible syntax (Requires pulp==3.3.2 in requirements.txt)
    food_vars = pulp.LpVariable.dicts("Food", valid_indices, lowBound=0, upBound=max_servings, cat='Integer')
    
    total_cals = pulp.lpSum([df.loc[i, 'Calories'] * food_vars[i] for i in valid_indices])
    total_p = pulp.lpSum([df.loc[i, 'Protein'] * food_vars[i] for i in valid_indices])
    total_f = pulp.lpSum([df.loc[i, 'Fat'] * food_vars[i] for i in valid_indices])
    total_c = pulp.lpSum([df.loc[i, 'Carbs'] * food_vars[i] for i in valid_indices])
    
    slack_cal = pulp.LpVariable("Slack_Cal", lowBound=0)
    slack_p = pulp.LpVariable("Slack_P", lowBound=0)
    
    prob += slack_cal + (slack_p * 10) 
    
    prob += total_cals - target_cals <= slack_cal
    prob += target_cals - total_cals <= slack_cal
    
    prob += total_p - target_p <= slack_p
    prob += target_p - total_p <= slack_p
    
    prob += total_f >= target_f - 10
    prob += total_f <= target_f + 10
    prob += total_c >= target_c - 15
    prob += total_c <= target_c + 15

    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        meal_items = df[df['Meal'] == meal].index.tolist()
        if len(meal_items) > 0:
            prob += pulp.lpSum([food_vars[i] for i in meal_items]) >= 1

    # This CBC command requires PuLP 3.x
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    results = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i in valid_indices:
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
    exclusions = st.text_input("Exclude ingredients (comma separated)", value="almonds, dairy")
    
url_map = {
    "Marciano": "https://www.bu.edu/dining/location/marciano/#menu",
    "West": "https://www.bu.edu/dining/location/west/#menu",
    "Warren": "https://www.bu.edu/dining/location/warren/#menu",
    "Fenway": "https://bufenway.sodexomyway.com/en-us/locations/the-fenway-dining-hall"
}

if st.button("Generate Meal Plan"):
    with st.spinner('Scraping live menu and running MILP solver...'):
        menu_df = scrape_dining_hall(url_map[dining_hall])
        
        if exclusions:
            bad_words = [word.strip().lower() for word in exclusions.split(',')]
            for word in bad_words:
                menu_df = menu_df[~menu_df['Ingredients'].str.contains(word, case=False, na=False)]
                
        plan_df = optimize_meals(menu_df, cals, protein, fat, carbs)
        
        if not plan_df.empty:
            st.success("Optimal Meal Plan Found!")
            
            for meal in ['Breakfast', 'Lunch', 'Dinner']:
                st.subheader(meal)
                meal_data = plan_df[plan_df['Meal'] == meal]
                st.table(meal_data[['Item', 'Servings', 'Calories', 'Protein']])
                
            st.metric("Total Planned Calories", plan_df['Calories'].sum())
            st.metric("Total Planned Protein", plan_df['Protein'].sum())
        else:
            st.error("Could not find a mathematical combination to hit those exact macros with today's menu. Try adjusting your targets.")
