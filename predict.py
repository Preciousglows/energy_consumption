import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("energy_consumption1.csv")


# Temperature-Humidity Index (THI) - Proxy for heat intensity
def compute_heat_index(min_temp, max_temp, humidity):
    avg_temp = (min_temp + max_temp) / 2
    return avg_temp + (0.2 * humidity)  # Simple heat index formula

# Apply new feature extraction
data["Heat_Index"] = data.apply(lambda row: compute_heat_index(row["Min_Temperature"], row["Max_Temperature"], row["Humidity"]), axis=1)

# Print sample
print(data.head())

# Features selection
features = ['Month', 'Year', 'Min_Temperature', 'Max_Temperature', 'Humidity', 'Wind_Speed', 'Rainfall', 'Solar_Radiation', 
            'Region', 'Heat_Index']
target = 'Energy_Consumption'
X = data[features]
y = data[target]

# Handle missing values by filling them with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# # Train Model
# model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
# model.fit(X_train, y_train)

#Initialize the XGBoost model with tuned hyperparameters
xgb_model = XGBRegressor(
    n_estimators=300, # Increase the number of trees
    learning_rate=0.05, #Reduce learning rate for better generalization
    max_depth=5, # Increase depth to capture complex relationships
    subsample=0.8, # Increase subsampling to reduce overfitting
    colsample_bytree=1.0, # Feature selction per tree
    objective="reg:squarederror",
    random_state=42
)

# xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

#GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)

# print("Best Hyperparameters:", grid_search.best_params_)
# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# def predict_energy(month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, region):
#     input_data = np.array([[month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, region]])
#     return model.predict(input_data)[0]

def predict_energy(month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, region):
    # Convert region name back to numerical code
    region_number = {v: k for k, v in region_mapping.items()}  # Reverse mapping
    region_code = region_number.get(region, None)  # Get the number, default to None if not found

    if region_code is None:
        raise ValueError(f"Invalid region name: {region}")  # Handle invalid input gracefully
    
    #Compute extra features
    heat_index = compute_heat_index(min_temp, max_temp, humidity)

    # Prepare input data for prediction
    input_data = np.array([[month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, region_code, heat_index]])
    
    return xgb_model.predict(input_data)[0]


# Streamlit UI
st.title("⚡ Energy Consumption Prediction")
st.write("Fill all the fields in the form below to get predictions.")

# Improved Form Layout
with st.expander("📌 **Enter Input Parameters**", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox("📅 Month", list(range(1, 13)), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
        year = st.selectbox("🗓 Year", list(range(2025, 2031)))
        # region_mapping = {"Abuja":1, "Benin":2, "Eko":3, "Port Harcourt":4, "Enugu":5, "Ibadan":6, "Ikeja":7, "Jos":8, "Yola":9, "Kano":10, "Kaduna":11}
        # Define the mapping of encoded region values to actual region names
        region_mapping = {
            1: "Abuja", 2: "Benin", 3: "Eko", 4: "Port Harcourt", 5: "Enugu",
            6: "Ibadan", 7: "Ikeja", 8: "Jos", 9: "Yola", 10: "Kano", 11: "Kaduna"
        }
        
        # region_name = st.selectbox("📍 Region", list(region_mapping.keys()))  
        # region = region_mapping[region_name]  # Convert to numeric code

        selected_region = st.selectbox("Select Region", list(region_mapping.values()))

    with col2:
        min_temp = st.slider("🌡 Min Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
        max_temp = st.slider("🔥 Max Temperature (°C)", min_value=0.0, max_value=50.0, value=30.0)
        humidity = st.slider("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        wind_speed = st.slider("🌬 Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=10.0)
        rainfall = st.slider("🌧 Rainfall (mm)", min_value=0.0, max_value=500.0, value=5.0)
        solar_radiation = st.slider("☀ Solar Radiation (W/m²)", min_value=0.0, max_value=2000.0, value=500.0)

if st.button("🔍 Predict Energy Consumption"):
    if all([month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, selected_region]):
        prediction = predict_energy(month, year, min_temp, max_temp, humidity, wind_speed, rainfall, solar_radiation, selected_region)


         # Show selected inputs
        st.markdown(f"""
        **🔹 User Selections:**  
        - 📅 **Month:** {month}  
        - 📆 **Year:** {year}  
        - 🌡️ **Min Temperature:** {min_temp}°C  
        - 🌡️ **Max Temperature:** {max_temp}°C  
        - 💧 **Humidity:** {humidity}%  
        - 🌬️ **Wind Speed:** {wind_speed} m/s  
        - 🌧️ **Rainfall:** {rainfall} mm  
        - ☀️ **Solar Radiation:** {solar_radiation} kWh/m²  
        - 📍 **Region:** {selected_region}  
        """)
        
        # Display Prediction
        st.success(f"#### ✅ **Predicted Energy Consumption:** {prediction:.2f} GWh")

        # Model Performance Metrics
        y_pred = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.write("### 📊 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "MAE", f"{mae:.2f}", 
            help="""📌 **MAE Performance**
               \n•  **Measures avg. error in GWh.**
                \n• **Excellent:** < 10 GWh
                \n• **Fair:** 10 - 20 GWh
                \n• **Poor:** > 20 GWh"""
        )

        col2.metric(
            "MSE", f"{mse:.2f}", 
            help="""📌 **MSE Performance**
                \n **Penalizes large errors.**
                \n• **Excellent:** < 200 GWh²
                \n• **Fair:** 200 - 500 GWh²
                \n• **Poor:** > 500 GWh²"""
        )

        col3.metric(
            "RMSE", f"{rmse:.2f}", 
            help="""📌 **RMSE Performance**
            \n•  **Measures error magnitude in GWh.**
            \n• **Excellent:** < 15 GWh
            \n• **Fair:** 15 - 25 GWh
            \n• **Poor:** > 25 GWh"""
        )

        col4.metric(
            "R² Score", f"{r2:.2f}", 
            help="""📌 **R² Score Performance**
            \n **Measures how well the model explains variance.**
            \n• **Excellent:** > 0.90
            \n• **Fair:** 0.80 - 0.90
            \n• **Poor:** < 0.80"""
        )

        # Feature Importance Visualization
        importances = xgb_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.write("### 🔥 Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This chart ranks the features based on their impact on energy consumption prediction.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Distribution of Energy Consumption
        st.write("### 📈 Energy Consumption Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Energy_Consumption'], kde=True, bins=30, ax=ax)
        ax.set_xlabel("Energy Consumption (GWh)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This histogram shows the frequency of different energy consumption levels, helping to detect patterns and outliers.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Correlation Heatmap
        st.write("### 🔗 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This heatmap visualizes the relationships between different features, helping to detect multicollinearity and key influencing factors.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Actual vs Predicted Plot
        st.write("### 📉 Actual vs. Predicted Energy Consumption")
        fig, ax = plt.subplots()
        ax.plot(y_test.values[:50], label="Actual", marker="o")
        ax.plot(y_pred[:50], label="Predicted", marker="x")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Energy Consumption (GWh)")
        ax.legend()
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This plot compares real energy values to the model's predictions. Closer alignment indicates higher accuracy.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


        # Boxplot by Month
        st.subheader("📦 Boxplot of Energy Consumption by Month")
        fig, ax = plt.subplots()
        sns.boxplot(x='Month', y='Energy_Consumption', data=data, ax=ax)
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This boxplot visualizes how energy consumption varies across different months, highlighting seasonal trends.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create Bar Chart
        # Replace encoded region numbers with actual names
        data['Region'] = data['Region'].map(region_mapping)

        # Create a color palette based on the number of unique regions
        region_colors = dict(zip(data['Region'].unique(), sns.color_palette("husl", len(data['Region'].unique()))))
        st.subheader("🏙️ Bar Chart of Regional Energy Consumption")
        fig, ax = plt.subplots()

        # Plot the bar chart with custom colors
        region_avg_energy = data.groupby('Region')['Energy_Consumption'].mean()

        # Plot with actual region names and corresponding colors
        region_avg_energy.plot(kind='bar', ax=ax, color=[region_colors[region] for region in region_avg_energy.index])


        # Labeling
        ax.set_ylabel("Average Energy Consumption")
        ax.set_xlabel("Region")
        ax.set_title("Average Energy Consumption by Region")
        ax.set_xticklabels(region_avg_energy.index, rotation=45)  # Rotate labels for clarity

        # Show Plot
        st.pyplot(fig)
        st.markdown(
            """
            <div style="text-align: center;">
                 <strong>This bar chart shows the average energy usage across different regions, 
                helping to compare demand variations.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.error("❌ Please fill in all fields!")

        