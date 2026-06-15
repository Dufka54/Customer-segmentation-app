🛍️ Mall Customer Segmentation Dashboard

An interactive, multi-dimensional exploratory dashboard built to perform customer segmentation on the Mall Customers dataset. This project avoids pre-packaged machine learning models (like scikit-learn) and instead relies on a custom, scratch-built $K$-means clustering algorithm developed purely with Python and vectorized NumPy structures.

📌 Dashboard Overview

The application features a four-tab interactive experience:

📈 Exploratory Data Analysis: Explore raw consumer profiles, check distributions, and inspect transformed scaling steps.

🤖 Interactive Clustering Visualizations: Run the custom $K$-means engine in real time with dynamic sliders for $k$ (clusters) and maximum epochs. Inspect interactive 2D and 3D Plotly scatter plots.

🎯 Customer Personas & Radar DNA: Profile each discovered segment automatically and view their normalized Radar Chart DNA across all 4 input dimensions. Export your tagged segments as a CSV.

📊 Grid Search & Metric Tuning: Perform a comprehensive grid search to isolate optimal parameters utilizing the Silhouette Score and Within-Cluster Sum of Squares (WCSS).

🧮 Mathematical Architecture

1. Vectorized Euclidean Spatial Distance

The absolute straight-line distance between any point $x$ and centroid $y$ across $N$-dimensions is computed using the classical Euclidean distance:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

Using vectorized np.sum and array broadcasting, the algorithm processes multi-dimensional input spaces (such as our 4D customer features) with high performance and minimal memory overhead.

2. Feature Normalization (MinMaxScaler)

To ensure features with larger magnitude ranges (such as annual income spanning up to $\$137k$) do not dominate features with smaller ranges (such as age or binary-encoded gender), we transform all features to a standardized scale of $[0, 1]$:

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

3. Geometric Cluster Quality Metrics

Silhouette Score Analysis

Used to evaluate the separation distance between the resulting clusters. The silhouette score for a single sample is defined as:

$$s = \frac{b - a}{\max(a, b)}$$

where $a$ is the mean intra-cluster distance, and $b$ is the mean nearest-cluster distance. Scores closer to $+1$ indicate highly dense, well-separated groupings.

Within-Cluster Sum of Squares (WCSS / Inertia)

Calculates the total squared distance from each observation to its assigned cluster centroid:

$$\text{WCSS} = \sum_{k=1}^{K}\sum_{x \in C_k} ||x - \mu_k||^2$$

