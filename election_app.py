import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# --- 1. Page Configuration and Creative Title ---
st.set_page_config(
    page_title="India Votes 2024: Election Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("India Votes 2024: The Grand Election Saga! 🗳️")
st.markdown(
    """
    Welcome to an interactive journey through the **2024 Indian General Election results!** This dashboard brings complex data to life, allowing you to explore key insights, 
    understand the numbers behind the mandate, and even dabble in predicting outcomes.
    
    _Let's dive into the democratic heartbeat of India!_
    """
)

st.markdown("---") # Separator for visual appeal

# --- Data Loading (Simulated for this example) ---
# In a real scenario, you would load your actual CSV/Excel file here.
# For demonstration, we'll create a synthetic dataset resembling election data.

@st.cache_data # Cache data to avoid reloading on every rerun
def load_and_prepare_data():
    data = {
        'Constituency': [f'C_{i:03d}' for i in range(1, 544)],
        'State': np.random.choice(['Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Bihar', 'Tamil Nadu', 'Karnataka', 'Rajasthan', 'Madhya Pradesh', 'Gujarat', 'Other States'], 543),
        'Winning_Party': np.random.choice(['BJP', 'INC', 'SP', 'TMC', 'DMK', 'Other'], 543, p=[0.45, 0.20, 0.08, 0.07, 0.05, 0.15]),
        'Total_Votes_Polled': np.random.randint(500000, 1500000, 543),
        'Winning_Candidate_Votes': np.random.randint(250000, 1000000, 543),
        'RunnerUp_Candidate_Votes': np.random.randint(100000, 500000, 543),
        'Voter_Turnout_Percentage': np.random.uniform(50, 85, 543).round(2),
        'Previous_Election_Margin_Avg': np.random.uniform(5, 25, 543).round(2), # Hypothetical past data
        'Candidate_Sentiment_Score': np.random.uniform(0.3, 0.9, 543).round(2), # Hypothetical ML feature
        'Development_Index': np.random.uniform(0.1, 1.0, 543).round(2) # Hypothetical ML feature
    }
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['Winning_Margin'] = df['Winning_Candidate_Votes'] - df['RunnerUp_Candidate_Votes']
    df['Vote_Share_Winning_Party'] = (df['Winning_Candidate_Votes'] / df['Total_Votes_Polled'] * 100).round(2)

    return df

df = load_and_prepare_data()

# --- Sidebar Navigation ---
st.sidebar.header("Explore the Election Data")
page_selection = st.sidebar.radio(
    "Go to:",
    ["📊 Dashboard Overview", "🧹 Data Deep Dive & Cleaning", "📈 Statistical Insights", "🤖 Predict the Winner!"]
)

# --- 2. Dashboard Overview ---
if page_selection == "📊 Dashboard Overview":
    st.header("Dashboard Overview: The Big Picture 🗺️")
    st.markdown(
        """
        Get a quick snapshot of the 2024 election results. See the overall seat distribution, 
        top parties, and key metrics at a glance.
        """
    )

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    total_constituencies = len(df)
    total_votes_polled = df['Total_Votes_Polled'].sum()
    avg_voter_turnout = df['Voter_Turnout_Percentage'].mean().round(2)

    col1.metric("Total Constituencies", f"{total_constituencies} Seats")
    col2.metric("Total Votes Polled", f"{total_votes_polled/10**8:.2f} Cr") # Display in Crores
    col3.metric("Avg. Voter Turnout", f"{avg_voter_turnout}%")

    st.markdown("---")

    st.subheader("Party-wise Seat Distribution")
    party_counts = df['Winning_Party'].value_counts().reset_index()
    party_counts.columns = ['Party', 'Seats Won']
    
    fig_seats = px.bar(
        party_counts,
        x='Party',
        y='Seats Won',
        title='Seats Won by Each Party',
        color='Party',
        template='plotly_white',
        labels={'Party': 'Political Party', 'Seats Won': 'Number of Seats'}
    )
    st.plotly_chart(fig_seats, use_container_width=True)

    st.subheader("Vote Share by Winning Party")
    party_vote_share = df.groupby('Winning_Party')['Total_Votes_Polled'].sum().reset_index()
    party_vote_share.columns = ['Party', 'Total_Votes_Polled']
    party_vote_share['Vote_Share_Percentage'] = (party_vote_share['Total_Votes_Polled'] / party_vote_share['Total_Votes_Polled'].sum() * 100).round(2)

    fig_vote_share = px.pie(
        party_vote_share,
        names='Party',
        values='Vote_Share_Percentage',
        title='Overall Vote Share by Winning Party',
        hole=0.4, # Donut chart
        template='plotly_white'
    )
    st.plotly_chart(fig_vote_share, use_container_width=True)


# --- 3. Data Deep Dive & Cleaning ---
elif page_selection == "🧹 Data Deep Dive & Cleaning":
    st.header("Data Deep Dive: Unveiling the Raw Numbers 🕵️‍♀️")
    st.markdown(
        """
        Explore the raw dataset and understand how we ensure its quality 
        through essential data cleaning steps.
        """
    )

    st.subheader("Raw Data Sample")
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    st.subheader("Initial Data Information")
    st.code(df.info())

    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.warning("Missing values detected! (Though simulated data has none)")
        st.table(missing_data.sort_values(ascending=False))
        st.markdown(
            """
            *Self-correction note: In a real scenario, you'd handle these using methods 
            like `df.dropna()` or `df.fillna()` based on the column type and context.*
            """
        )
    else:
        st.success("No missing values in this simulated dataset! 🎉")

    st.subheader("Data Type Conversion & Consistency (Demonstration)")
    st.markdown(
        """
        Here's how we might ensure data types are correct and consistent. 
        For instance, ensuring vote counts are integers and percentages are floats.
        (Our simulated data is already set, but this shows the intent.)
        """
    )
    # Example of type conversion (even if not strictly needed for this simulated data)
    df['Total_Votes_Polled'] = df['Total_Votes_Polled'].astype(int)
    df['Winning_Candidate_Votes'] = df['Winning_Candidate_Votes'].astype(int)
    st.info("Ensured 'Total_Votes_Polled' and 'Winning_Candidate_Votes' are integers.")


# --- 4. Statistical Insights ---
elif page_selection == "📈 Statistical Insights":
    st.header("Statistical Insights: Decoding the Mandate 🧠")
    st.markdown(
        """
        Dive into the numbers! Discover key statistical features and interactive visualizations 
        that reveal patterns and anomalies in the election results.
        """
    )

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Top/Bottom Performing Constituencies")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Top 10 Constituencies by Voter Turnout")
        top_turnout = df.nlargest(10, 'Voter_Turnout_Percentage')[['Constituency', 'State', 'Voter_Turnout_Percentage']]
        st.dataframe(top_turnout)
    with col2:
        st.markdown("##### Bottom 10 Constituencies by Voter Turnout")
        bottom_turnout = df.nsmallest(10, 'Voter_Turnout_Percentage')[['Constituency', 'State', 'Voter_Turnout_Percentage']]
        st.dataframe(bottom_turnout)

    st.markdown("---")

    st.subheader("Voter Turnout Distribution")
    fig_turnout = px.histogram(
        df,
        x='Voter_Turnout_Percentage',
        nbins=20,
        title='Distribution of Voter Turnout Across Constituencies',
        labels={'Voter_Turnout_Percentage': 'Voter Turnout (%)'},
        template='plotly_white'
    )
    st.plotly_chart(fig_turnout, use_container_width=True)

    st.subheader("Winning Margin Analysis")
    fig_margin = px.histogram(
        df,
        x='Winning_Margin',
        nbins=30,
        title='Distribution of Winning Margins',
        labels={'Winning_Margin': 'Vote Difference (Winner vs. Runner-up)'},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='plotly_white'
    )
    st.plotly_chart(fig_margin, use_container_width=True)

    st.subheader("Party Performance by State (Interactive)")
    selected_state = st.selectbox(
        "Select a State to view Party Performance:",
        options=sorted(df['State'].unique())
    )
    state_df = df[df['State'] == selected_state]
    party_state_counts = state_df['Winning_Party'].value_counts().reset_index()
    party_state_counts.columns = ['Party', 'Seats Won']

    fig_state_party = px.bar(
        party_state_counts,
        x='Party',
        y='Seats Won',
        title=f'Seats Won by Party in {selected_state}',
        color='Party',
        template='plotly_white'
    )
    st.plotly_chart(fig_state_party, use_container_width=True)


# --- 5. Machine Learning Prediction ---
elif page_selection == "🤖 Predict the Winner!":
    st.header("Predict the Winner: The ML Crystal Ball 🔮")
    st.markdown(
        """
        Ever wondered what factors influence an election outcome? Here, we use a
        simple Machine Learning model to predict the **Winning Party** for a hypothetical constituency
        based on a few key features.
        
        **Disclaimer:** This is a simplified demonstration using synthetic data. Real-world election 
        prediction is far more complex and involves vast amounts of diverse data and sophisticated models.
        """
    )

    st.subheader("Model Training")
    st.write("We'll train a Decision Tree Classifier on our simulated election data.")
    
    # Define features (X) and target (y)
    features = ['Total_Votes_Polled', 'Voter_Turnout_Percentage', 
                'Previous_Election_Margin_Avg', 'Candidate_Sentiment_Score', 'Development_Index']
    target = 'Winning_Party'

    # Encode categorical target variable if necessary (Decision Tree handles string labels directly, but good practice for other models)
    # y = df[target] # For Decision Tree, string labels work directly
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Machine Learning Model (Decision Tree) Trained Successfully!")
    st.write(f"Model Accuracy on Test Data: **{accuracy:.2f}** (This is for illustrative purposes with simulated data)")
    
    with st.expander("See Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    st.markdown("---")

    st.subheader("Predict for a Hypothetical Constituency")

    # User inputs for prediction
    st.sidebar.subheader("Adjust Features for Prediction")
    
    input_total_votes = st.sidebar.slider("Total Votes Polled:", int(df['Total_Votes_Polled'].min()), int(df['Total_Votes_Polled'].max()), int(df['Total_Votes_Polled'].mean()))
    input_voter_turnout = st.sidebar.slider("Voter Turnout Percentage:", float(df['Voter_Turnout_Percentage'].min()), float(df['Voter_Turnout_Percentage'].max()), float(df['Voter_Turnout_Percentage'].mean()))
    input_prev_margin = st.sidebar.slider("Previous Election Margin (Avg. %):", float(df['Previous_Election_Margin_Avg'].min()), float(df['Previous_Election_Margin_Avg'].max()), float(df['Previous_Election_Margin_Avg'].mean()))
    input_sentiment_score = st.sidebar.slider("Candidate Sentiment Score (0.0-1.0):", 0.0, 1.0, float(df['Candidate_Sentiment_Score'].mean()), 0.01)
    input_development_index = st.sidebar.slider("Local Development Index (0.0-1.0):", 0.0, 1.0, float(df['Development_Index'].mean()), 0.01)

    predict_button = st.button("Predict Winning Party")

    if predict_button:
        input_data = pd.DataFrame([[input_total_votes, input_voter_turnout, 
                                     input_prev_margin, input_sentiment_score, 
                                     input_development_index]], 
                                    columns=features)
        
        predicted_party = model.predict(input_data)[0]
        
        st.markdown(f"### 🎉 Based on your inputs, the predicted winning party is: **{predicted_party}**")
        st.balloons()

        st.markdown(
            """
            *Remember, this prediction is purely illustrative. Election outcomes are influenced by a myriad of complex, 
            often unpredictable, real-world factors beyond simple numerical inputs.*
            """
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: small; color: gray;">
        Built with ❤️ using Streamlit for Data Enthusiasts.
    </div>
    """,
    unsafe_allow_html=True
)
