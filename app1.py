import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components  # For GTM

# Function to inject GTM
def inject_gtm(gtm_id):
    # GTM Script
    gtm_script = f"""
    <!-- Google Tag Manager -->
    <script>
    (function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{'gtm.start':
    new Date().getTime(),event:'gtm.js'}});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    }})(window,document,'script','dataLayer','{gtm_id}');
    </script>
    <!-- End Google Tag Manager -->
    """
    
    # GTM Noscript
    gtm_noscript = f"""
    <!-- Google Tag Manager (noscript) -->
    <noscript>
      <iframe src="https://www.googletagmanager.com/ns.html?id={gtm_id}"
      height="0" width="0" style="display:none;visibility:hidden"></iframe>
    </noscript>
    <!-- End Google Tag Manager (noscript) -->
    """
    
    # Inject GTM Script
    components.html(gtm_script, height=0, width=0)
    
    # Inject GTM Noscript
    components.html(gtm_noscript, height=0, width=0)

# Set page configuration
st.set_page_config(page_title="Data Analyse App", layout="wide")

# Inject GTM
inject_gtm('GTM-5N2NNDML')  # Your GTM ID

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Visualization", "Machine Learning", "About"])

# Function to display the Home page
def home():
    st.title("Advanced Streamlit App")
    st.write("Welcome to the Data Analyse app. Use the sidebar to navigate through different sections.")

    # Personalized Greeting
    st.subheader("Personalized Greeting")
    name = st.text_input("Enter your name:")
    if name:
        st.write(f"Hello, **{name}**! Welcome to the app.")

    # Age slider and feedback
    st.subheader("Age Information")
    age = st.slider("Select your age:", 0, 100, 25)
    st.write(f"Your age is **{age}**.")

    if age < 18:
        st.warning("You are a minor.")
    elif 18 <= age < 65:
        st.success("You are an adult.")
    else:
        st.info("You are a senior.")

# Function to display the Data Analysis page
def data_analysis():
    st.title("Data Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Data", df.head())

        # Handle missing values
        st.write("#### Handling Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(df.isnull())

        if st.button("Fill Missing Values with Mean"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.write("Missing values filled with mean:", df)

        # Display descriptive statistics
        st.write("#### Descriptive Statistics")
        st.write(df.describe())

        # Option to download the cleaned data
        st.download_button(
            label="Download Cleaned Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_data.csv',
            mime='text/csv',
        )
    else:
        st.info("Please upload a CSV or Excel file to proceed.")

# Function to display the Visualization page
def visualization():
    st.title("Data Visualization")
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file for visualization", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Data", df.head())

        # Sidebar for plot selection
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap"])

        if plot_type == "Histogram":
            st.subheader("Histogram")
            column = st.selectbox("Select column for Histogram", df.select_dtypes(include=[np.number]).columns)
            bins = st.slider("Number of Bins", min_value=5, max_value=100, value=30)
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color='skyblue')
            st.pyplot(fig)

        elif plot_type == "Boxplot":
            st.subheader("Boxplot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            column = st.selectbox("Select column for Boxplot", numeric_cols)
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=df[column], ax=ax2, color='lightgreen')
            st.pyplot(fig2)

        elif plot_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            hue = st.selectbox("Color By (Optional)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())

            fig3, ax3 = plt.subplots()
            if hue != "None" and hue is not None:
                sns.scatterplot(x=x_col, y=y_col, hue=hue, data=df, ax=ax3)
            else:
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax3, color='orange')
            st.pyplot(fig3)

        elif plot_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig4, ax4 = plt.subplots(figsize=(10,8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
            st.pyplot(fig4)
    else:
        st.info("Please upload a CSV or Excel file to proceed.")

# Function to display the Machine Learning page
def machine_learning():
    st.title("Machine Learning - Linear Regression")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file for ML", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Data", df.head())

        # Select target and features
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Feature(s)", [col for col in df.columns if col != target])

        if len(features) < 1:
            st.warning("Please select at least one feature.")
        else:
            # Handle missing values
            if st.checkbox("Show Missing Values"):
                st.write(df[features + [target]].isnull())

            if st.button("Fill Missing Values with Mean"):
                df[features] = df[features].fillna(df[features].mean())
                df[target] = df[target].fillna(df[target].mean())
                st.write("Missing values filled with mean:", df)

            # Encode categorical variables
            df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object', 'category']).columns, drop_first=True)
            X = df_encoded[features]
            y = df_encoded[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("### Model Performance")
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.2f}")

            # Display coefficients
            coefficients = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_
            })
            st.write("### Model Coefficients")
            st.write(coefficients)

            # Option to download the model
            if st.button("Download Model"):
                joblib.dump(model, 'linear_regression_model.pkl')
                with open('linear_regression_model.pkl', 'rb') as f:
                    st.download_button(
                        label="Download Linear Regression Model",
                        data=f,
                        file_name='linear_regression_model.pkl',
                        mime='application/octet-stream',
                    )
    else:
        st.info("Please upload a CSV or Excel file to proceed.")

# Function to display the About page
def about():
    st.title("About")
    col1, col2 = st.columns(2)

    with col1:
        st.header("App Information")
        st.write("""
        This advanced Streamlit app demonstrates various features including data analysis, visualization, and interactive widgets.

        **Features:**
        - Multiple Pages Navigation
        - Data Cleaning and Handling Missing Values
        - Descriptive Statistics
        - Interactive Data Visualizations with Seaborn and Matplotlib
        - Machine Learning Integration (Linear Regression)
        - User Inputs and Interactions
        """)

    with col2:
        st.header("Technologies Used")
        st.write("""
        - Streamlit
        - Pandas
        - NumPy
        - Seaborn
        - Matplotlib
        - Scikit-learn
        - Pillow
        """)

    # Adding an image
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)

# Routing based on user selection
if page == "Home":
    home()
elif page == "Data Analysis":
    data_analysis()
elif page == "Visualization":
    visualization()
elif page == "Machine Learning":
    machine_learning()
elif page == "About":
    about()
