import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="AI Customer Segmenter",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .dark .metric-card {
        background-color: #1a202c;
        border: 1px solid #2d3748;
    }
    </style>
""", unsafe_allow_html=True)

class kmeans:
    def __init__(self, cluster, iterations):
        self.cluster = cluster
        self.iterations = iterations
        self.centroid = None

    def fit(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Safety Check: Check if self.cluster is larger than the dataset
        if self.cluster > X.shape[0]:
            self.cluster = X.shape[0]

        rnd_idx = random.sample(range(0, X.shape[0]), self.cluster)
        self.centroid = X.iloc[rnd_idx].values
        
        cluster_grp = np.zeros(X.shape[0])
        for i in range(self.iterations):
            cluster_grp = self.assignCluster(X)
            curr_centroid = self.centroid.copy()
            self.centroid = self.new_centroid(X, cluster_grp)
            if curr_centroid.shape == self.centroid.shape and (curr_centroid == self.centroid).all():
                break
                
        return cluster_grp

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def assignCluster(self, X):
        cluster_grp = []
        X_vals = X.values 
        
        for r in range(X_vals.shape[0]):
            dist = []
            for center in self.centroid:
                dist.append(self.euclidean_distance(X_vals[r], center))
            min_dist = min(dist)
            idx_pos = dist.index(min_dist)
            cluster_grp.append(idx_pos)
        return np.array(cluster_grp)

    def new_centroid(self, X, cluster_grp):
        new_centroids = []
        for i in range(self.cluster):
            members = X[cluster_grp == i]
            if len(members) > 0:
                new_centroids.append(members.mean(axis=0))
            else:
                new_centroids.append(self.centroid[i])
        return np.array(new_centroids)

# Helper function to compute Inertia / Within-Cluster Sum of Squares (WCSS)
def compute_wcss(X, centroids, labels):
    X_vals = X.values
    wcss = 0.0
    for idx, row in enumerate(X_vals):
        cluster_idx = labels[idx]
        centroid = centroids[cluster_idx]
        wcss += np.sum((row - centroid) ** 2)
    return wcss

@st.cache_data
def load_mall_customers_dataset():
    filename = 'Mall_Customers.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        st.error(f"⚠️ Could not locate '{filename}' in the workspace directory. Creating a quick fallback dataframe.")
        return pd.DataFrame({
            'CustomerID': range(1, 11),
            'Gender': ['Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female'],
            'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
            'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
            'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
        })


st.markdown("<h1 class='main-header'>Mall Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Perform high-dimensional consumer profiling using a scratch-built K-means implementation.</p>", unsafe_allow_html=True)

st.sidebar.header("📁 Data Source & Parameters")

data_source = st.sidebar.radio("Data Input Choice:", ["Use Pre-loaded Mall_Customers.csv", "Upload Custom CSV File"])

if data_source == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        required_cols = {'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'}
        if not required_cols.issubset(raw_df.columns):
            st.sidebar.error(f"CSV must contain these columns: {required_cols}")
            raw_df = load_mall_customers_dataset()
    else:
        st.sidebar.info("Awaiting file upload. Using default Mall_Customers.csv dataset.")
        raw_df = load_mall_customers_dataset()
else:
    raw_df = load_mall_customers_dataset()

st.sidebar.subheader("⚙️ Preprocessing Pipelines")
use_custom_spending_bins = st.sidebar.toggle("Enable Custom Spending score Category Mapping", value=True, 
                                            help="Divides raw continuous spending score into 4 ordered bins (Low, Med-Low, Med-High, High) and LabelEncodes them.")

all_features = ['Age', 'Annual Income (k$)', 'Gender_Encoded']
if use_custom_spending_bins:
    all_features.append('Spending_Score_Encoded')
else:
    all_features.append('Spending Score (1-100)')

selected_features = st.sidebar.multiselect(
    "Choose dimensions for K-means:",
    options=all_features,
    default=all_features
)

st.sidebar.subheader("🧩 Hyperparameters")
k_val = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)
epochs_val = st.sidebar.slider("Max Iterations (epochs)", min_value=5, max_value=200, value=50, step=5)


@st.cache_data
def preprocess_dataset(df, use_bins):
    df_proc = df.copy()

    df_proc['Gender_Encoded'] = df_proc['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    
    if use_bins:
        df_proc['Spending_Category'] = pd.cut(df_proc['Spending Score (1-100)'], bins=[-1, 25, 50, 75, 101], labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        category_map = {'Low': 0, 'Medium-Low': 1, 'Medium-High': 2, 'High': 3}
        df_proc['Spending_Score_Encoded'] = df_proc['Spending_Category'].map(category_map)
        target_spending = 'Spending_Score_Encoded'
    else:
        target_spending = 'Spending Score (1-100)'
        
    features_to_scale = ['Age', 'Annual Income (k$)', target_spending, 'Gender_Encoded']
    scaler = MinMaxScaler()
    
    df_scaled_vals = scaler.fit_transform(df_proc[features_to_scale])
    df_scaled = pd.DataFrame(df_scaled_vals, columns=features_to_scale)
    
    return df_proc, df_scaled, features_to_scale

df_proc, df_scaled, scaled_cols = preprocess_dataset(raw_df, use_custom_spending_bins)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exploratory Analysis & Preprocessing",
    "🤖 Interactive Clustering Visualizations",
    "🎯 Customer Personas & Radar DNA",
    "📊 Grid Search & Metric Tuning"
])

with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Raw Consumer Records")
        st.dataframe(raw_df, use_container_width=True)
        st.write(f"Total Records found: **{len(raw_df)} rows**")
        
    with col2:
        st.subheader("Engineered & MinMaxScaler Scaled Features")
        st.dataframe(df_scaled, use_container_width=True)
        st.caption("All features mapped to range [0, 1] to prevent distance calculations from being skewed by different measurement scales.")

    st.markdown("---")
    st.subheader("Raw Distributions")
    fig_hist = px.histogram(raw_df, x="Annual Income (k$)", color="Gender", marginal="box", 
                            title="Distribution of Annual Income by Gender", 
                            color_discrete_sequence=["#667eea", "#ec4899"])
    st.plotly_chart(fig_hist, use_container_width=True)


with tab2:
    st.header("Interactive Custom K-means Engine")
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features in the sidebar to run the clustering model.")
    else:
        X_train = df_scaled[selected_features]
        
        with st.spinner("Executing your custom K-means class..."):
            model = kmeans(cluster=k_val, iterations=epochs_val)
            labels = model.fit(X_train)
            
            df_proc['Cluster'] = labels
            df_scaled['Cluster'] = labels
            
        st.success(f"Successfully converged with {k_val} clusters!")

        col_viz1, col_viz2 = st.columns([1, 1])
        
        with col_viz1:
            st.subheader("2D Consumer Segments (Income vs Spend)")
            fig_2d = px.scatter(
                df_proc, 
                x="Annual Income (k$)", 
                y="Spending Score (1-100)", 
                color=df_proc["Cluster"].astype(str),
                symbol="Gender",
                hover_data=["Age"],
                title="Consumer Groups: Annual Income vs Spending Score",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_2d.update_layout(legend_title="Cluster ID")
            st.plotly_chart(fig_2d, use_container_width=True)
            st.caption("💡 Notice the distinct segments: Low Spenders vs. High Spenders in different income brackets.")
            
        with col_viz2:
            st.subheader("3D High-Dimensional Segments")
            fig_3d = px.scatter_3d(
                df_proc,
                x="Age",
                y="Annual Income (k$)",
                z="Spending Score (1-100)",
                color=df_proc["Cluster"].astype(str),
                title="Multi-dimensional Clusters (Age, Income, Spend)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_3d.update_layout(legend_title="Cluster ID")
            st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("Consumer Persona Profiles")
    
    if 'Cluster' in df_proc.columns:
        averages = df_proc.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
        
        counts = df_proc['Cluster'].value_counts().reset_index().rename(columns={'count': 'Customer Count'})
        profile_summary = pd.merge(averages, counts, on='Cluster')
        
        def assign_business_label(row):
            inc = row['Annual Income (k$)']
            spd = row['Spending Score (1-100)']
            age = row['Age']
            
            if inc > 70 and spd > 65:
                return "🎯 High Income, Heavy Spenders (Target Group)"
            elif inc > 70 and spd < 35:
                return "💼 Affluent Savers (High Income, Low Spend)"
            elif inc < 45 and spd > 65:
                return "💸 Impulsive Shoppers (Low Income, High Spend)"
            elif inc < 45 and spd < 35:
                return "🛑 Cost-Conscious Shoppers (Low Income, Low Spend)"
            else:
                if age > 45:
                    return "👴 Senior Average Value Segment"
                else:
                    return "🌱 Young Standard Consumers"
                    
        profile_summary['Strategic Segment Name'] = profile_summary.apply(assign_business_label, axis=1)
        
        st.subheader("Segment Breakdown")
        st.dataframe(
            profile_summary.style.format({
                'Age': '{:.1f} yrs',
                'Annual Income (k$)': '${:.1f}k',
                'Spending Score (1-100)': '{:.1f}/100',
                'Customer Count': '{:.0f} customers'
            }).highlight_max(subset=['Annual Income (k$)', 'Spending Score (1-100)'], color='#c6f6d5')
              .highlight_min(subset=['Annual Income (k$)', 'Spending Score (1-100)'], color='#fed7d7'),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Multi-Dimensional Persona DNA (Radar Chart)")
        
        radar_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
        radar_data = df_scaled.groupby('Cluster')[radar_cols].mean().reset_index()
        
        fig_radar = go.Figure()
        
        for index, row in radar_data.iterrows():
            cluster_id = int(row['Cluster'])
            r_values = [row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)'], row['Gender_Encoded']]
            r_values.append(r_values[0])
            categories = ['Age Index', 'Income Index', 'Spending Score Index', 'Gender (Females Ratio)', 'Age Index']
            
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories,
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Interactive Cluster DNA Comparison (Scaled Axes)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📥 Export Tagged Customer Data")

        output_df = raw_df.copy()
        output_df['Assigned_Cluster'] = df_proc['Cluster']
        
        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Segmented Data (CSV)",
            data=csv_data,
            file_name="segmented_mall_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("Run clustering in Tab 2 to view cluster persona profiles.")


with tab4:
    st.header("Model Optimization & Parameter Grid Search")
    
    st.markdown("""
        To evaluate model accuracy in **unsupervised clustering**, we measure:
        - **Silhouette Score:** Quantifies how similar objects are to their own cluster compared to neighboring clusters (Ranges from -1 to +1). High score represents tight, distinct boundaries.
        - **WCSS (Within-Cluster Sum of Squares / Inertia):** Measures how compact the clusters are.
    """)
    
    run_gs = st.button("🚀 Run Comprehensive Parameter Grid Search")
    
    if run_gs and len(selected_features) >= 2:
        k_values = [3, 5, 7, 9]
        epochs = [10, 50, 100]
        
        gs_results = []
        
        with st.spinner("Tuning hyperparameters (Performing Grid Search over combinations)..."):
            for k_candidate in k_values:
                for ep_candidate in epochs:
                    gs_model = kmeans(cluster=k_candidate, iterations=ep_candidate)
                    gs_labels = gs_model.fit(df_scaled[selected_features])
                    
                    if len(np.unique(gs_labels)) > 1:
                        score = silhouette_score(df_scaled[selected_features], gs_labels)
                        inertia = compute_wcss(df_scaled[selected_features], gs_model.centroid, gs_labels)
                    else:
                        score = -1.0
                        inertia = float('inf')
                        
                    gs_results.append({
                        'K (Clusters)': k_candidate,
                        'Epochs (Iterations)': ep_candidate,
                        'Silhouette Score': score,
                        'WCSS (Inertia)': inertia
                    })
                    
        gs_df = pd.DataFrame(gs_results)

        best_run = gs_df.loc[gs_df['Silhouette Score'].idxmax()]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🏆 Optimal K", f"{int(best_run['K (Clusters)'])}")
        with col_m2:
            st.metric("⏳ Optimal Max Epochs", f"{int(best_run['Epochs (Iterations)'])}")
        with col_m3:
            st.metric("🎯 Best Silhouette Score", f"{best_run['Silhouette Score']:.4f}")
            
        st.subheader("Grid Search Matrix (Silhouette Scores)")
        pivot_df = gs_df.pivot(index='K (Clusters)', columns='Epochs (Iterations)', values='Silhouette Score')
        st.dataframe(pivot_df.style.background_gradient(cmap='Purples'), use_container_width=True)
        
        st.subheader("The Elbow Method & Silhouette Trade-off Curve")
        elbow_df = gs_df.groupby('K (Clusters)').mean(numeric_only=True).reset_index()
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_elbow = px.line(elbow_df, x='K (Clusters)', y='WCSS (Inertia)', markers=True, 
                                title="Inertia Curve (Elbow Method)", color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig_elbow, use_container_width=True)
        with col_p2:
            fig_sil = px.line(elbow_df, x='K (Clusters)', y='Silhouette Score', markers=True,
                              title="Silhouette Score Over Cluster Range (K)", color_discrete_sequence=["#ec4899"])
            st.plotly_chart(fig_sil, use_container_width=True)
            
    elif len(selected_features) < 2:
        st.warning("⚠️ Make sure you have at least 2 features selected in your sidebar to calculate clustering metrics.")
    else:
        st.info("Click the button above to execute a systematic grid search across potential values of K and Training Iterations.")