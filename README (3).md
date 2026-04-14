# A Framework for Instagram Reach Analysis and Social Media Behavioral Analysis Using Machine Learning

## Overview

This repository contains two interconnected analytical modules built around Instagram data:

**Part 1 — Instagram Reach Analysis**: A machine learning pipeline that processes Instagram post metrics, models engagement patterns, and predicts post reach using regression and classification models.

**Part 2 — Social Media Behavioral Analysis (Mental Health and Suicidal Ideation Detection)**: A separate NLP-driven module that analyzes social media post content to classify mental health crisis signals, assess suicidal ideation risk levels, and generate crisis heatmaps.

Together, the two modules demonstrate how social media data can be analyzed both for content performance optimization and for identifying indicators of psychological distress in online behavior.

---

## Table of Contents

- [Part 1: Instagram Reach Analysis](#part-1-instagram-reach-analysis)
  - [Overview](#part-1-overview)
  - [Dataset](#part-1-dataset)
  - [Pipeline](#part-1-pipeline)
  - [Models Used](#part-1-models-used)
  - [Results](#part-1-results)
- [Part 2: Social Media Behavioral Analysis](#part-2-social-media-behavioral-analysis)
  - [Overview](#part-2-overview)
  - [Dataset](#part-2-dataset)
  - [Pipeline](#part-2-pipeline)
  - [Models Used](#part-2-models-used)
  - [Results](#part-2-results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Contributing](#contributing)
- [License](#license)

---

## Part 1: Instagram Reach Analysis

### Part 1 Overview

This module takes raw Instagram Insights data and applies data augmentation, exploratory analysis, NLP visualization, and machine learning to understand and predict post reach. The goal is to identify which engagement signals most strongly drive impressions and to build a model that can forecast reach for future posts.

### Part 1 Dataset

The input file is `reach analysis.csv`, which contains per-post Instagram metrics:

| Field           | Description                                              |
|-----------------|----------------------------------------------------------|
| Impressions     | Total number of times a post was displayed               |
| From Home       | Impressions from the follower home feed                  |
| From Hashtags   | Impressions driven by hashtag discovery                  |
| From Explore    | Impressions from the Explore page                        |
| From Other      | Impressions from other sources                           |
| Likes           | Number of likes received                                 |
| Comments        | Number of comments                                       |
| Shares          | Number of times the post was shared                      |
| Saves           | Number of times the post was saved                       |
| Profile Visits  | Profile visits attributed to the post                    |
| Follows         | New followers gained from the post                       |
| Caption         | Text caption of the post                                 |
| Hashtags        | Hashtags included in the post                            |

### Part 1 Pipeline

1. **Data Loading and Inspection**: Load `reach analysis.csv` with latin1 encoding; inspect shape, types, and missing values.
2. **Data Augmentation**: Gaussian noise is added to numerical columns to synthetically expand the dataset toward a target size of 1,200 rows. Hashtags are shuffled per augmented row to introduce textual variation while captions are kept intact.
3. **Exploratory Data Analysis**:
   - Distribution plots of impressions from Home and from Hashtags.
   - Donut pie chart breaking down total impressions by source (Home, Hashtags, Explore, Other).
   - Scatter plots with OLS trendlines for Likes vs. Impressions, Comments vs. Impressions, and Shares vs. Impressions.
   - Correlation matrix sorted by relationship strength to Impressions.
4. **NLP Visualization**:
   - Word cloud generated from all post captions to surface recurring content themes.
   - Word cloud generated from all hashtags to identify high-frequency tags.
5. **Conversion Rate Analysis**: Calculates the ratio of new followers to profile visits as a follower conversion metric.
6. **Reach Classification**: Posts are labelled Low (under 3,000 impressions), Medium (3,000–5,000), or High (above 5,000). Labels are encoded with `LabelEncoder` for model input.
7. **Feature Scaling**: `StandardScaler` is applied to normalize input features before classification.

### Part 1 Models Used

**Passive Aggressive Regressor** — Predicts continuous impression counts from engagement features (Likes, Saves, Comments, Shares, Profile Visits, Follows). Hyperparameter tuning is performed via `GridSearchCV` over regularization strength (`C`), intercept fitting, and max iterations. 5-fold cross-validation is used for evaluation, and MSE and R-squared are reported.

**Support Vector Machine (SVC, linear kernel)** — Classifies posts into Low, Medium, or High reach categories based on the same engagement features. Evaluation includes accuracy score, full classification report, and confusion matrix.

### Part 1 Results

- Saves and Shares are more strongly correlated with Impressions than Likes or Comments.
- Hashtag-sourced impressions follow a right-skewed distribution, indicating that a small subset of posts receive disproportionate hashtag-driven reach.
- The Passive Aggressive Regressor, after hyperparameter tuning and cross-validation, produces competitive R-squared scores with minimal training overhead.
- The SVM classifier achieves strong accuracy in separating Low, Medium, and High reach posts using only engagement-based features.

---

## Part 2: Social Media Behavioral Analysis

### Part 2 Overview

This module analyzes the text content of social media posts to detect mental health distress signals and classify posts by suicidal ideation risk level. The analysis runs across three sequential test phases, each producing a distinct output artifact. The intended application is early, automated identification of at-risk individuals based on their public posting behavior on platforms such as Instagram.

### Part 2 Dataset

This module operates on a corpus of social media posts that contain mental health-related content. Each record includes the raw post text, which is processed through NLP pipelines to extract risk-relevant linguistic signals. The classified output — covering 222 post records — is stored in `classified_mental_health_posts(Test1_Results).csv`.

### Part 2 Pipeline

1. **Text Preprocessing**: Post text is cleaned, tokenized, and normalized using spaCy and NLTK. Stop words are removed and relevant linguistic features are extracted.
2. **Mental Health and Suicidal Ideation Classification (Test 1)**: Each post is classified based on the presence and intensity of mental health distress or suicidal ideation signals. The output CSV contains the post text, assigned risk label, and supporting classification metadata for all 222 records.
3. **Crisis Risk Level Visualization (Test 2)**: Classified posts are grouped by risk tier and rendered as a distribution graph, showing the volume of posts at each level — from low concern through to crisis-level — giving a population-level view of mental health signal distribution.
4. **Crisis Heatmap Generation (Test 3)**: An interactive HTML heatmap is generated to visualize the density and spread of crisis-level posts across the dataset, supporting pattern identification across time or content categories.

### Part 2 Models Used

NLP-based classification using keyword extraction, sentiment analysis, and linguistic pattern matching implemented through spaCy and NLTK. The full implementation is contained in `social media behaviorl analysis.ipynb`, which spans 1,742 lines covering all three test phases end-to-end.

### Part 2 Results

- Test 1 produced a fully labelled dataset of 222 posts with per-post mental health and suicidal ideation risk classifications.
- Test 2 produced a static visualization revealing the distribution of posts across risk levels, showing the relative prevalence of low, moderate, high, and crisis-level content in the analyzed corpus.
- Test 3 produced an interactive HTML heatmap (`crisis_heatmap`) that renders crisis signal density, enabling visual pattern recognition across the dataset.

---

## Project Structure

```
a-framework-for-instagram-reach-analysis-using-ml/
│
├── a framework for instagram reach anlysis using ml.py         # Part 1: Reach analysis and ML pipeline
├── social media behaviorl analysis.ipynb                       # Part 2: Behavioral and suicidal ideation analysis
│
├── reach analysis.csv                                          # Input dataset for Part 1 (supply your own)
│
├── classified_mental_health_posts(Test1_Results).csv           # Part 2, Test 1 output: classified posts (222 records)
├── mental health crisis risk level graph(Test2_Results).png    # Part 2, Test 2 output: risk level distribution chart
├── crisis_heatmap(Test3_Results).html                          # Part 2, Test 3 output: interactive crisis heatmap
│
└── README.md
```

---

## Technologies Used

| Category                    | Libraries / Tools                                                                              |
|-----------------------------|------------------------------------------------------------------------------------------------|
| Data Processing             | Python, Pandas, NumPy                                                                          |
| Data Augmentation           | NumPy (Gaussian noise injection), custom row-level augmentation logic                          |
| Visualization               | Matplotlib, Seaborn, Plotly Express                                                            |
| Natural Language Processing | spaCy, NLTK, WordCloud                                                                         |
| Machine Learning            | Scikit-learn (PassiveAggressiveRegressor, SVC, GridSearchCV, StandardScaler, LabelEncoder)     |
| Notebook Environment        | Jupyter Notebook                                                                               |
| Output Formats              | CSV, PNG, Interactive HTML                                                                     |

---

## Installation

Python 3.8 or higher is required.

```bash
# Clone the repository
git clone https://github.com/tejaswantt/a-framework-for-instagram-reach-analysis-using-ml.git
cd a-framework-for-instagram-reach-analysis-using-ml

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy wordcloud plotly jupyter

# Download the spaCy English language model
python -m spacy download en_core_web_sm
```

---

## Usage

**Part 1 — Instagram Reach Analysis:**

Ensure `reach analysis.csv` is present in the project root, then run:

```bash
python "a framework for instagram reach anlysis using ml.py"
```

**Part 2 — Social Media Behavioral Analysis:**

```bash
jupyter notebook "social media behaviorl analysis.ipynb"
```

Run all cells in order. The notebook executes all three test phases sequentially and writes the three output files to the project directory.

---

## Output Files

| File | Part | Description |
|------|------|-------------|
| `classified_mental_health_posts(Test1_Results).csv` | Part 2 — Test 1 | Per-post mental health and suicidal ideation classification labels for 222 posts |
| `mental health crisis risk level graph(Test2_Results).png` | Part 2 — Test 2 | Static chart showing post volume by crisis risk level |
| `crisis_heatmap(Test3_Results).html` | Part 2 — Test 3 | Interactive heatmap of crisis signal density across the dataset |

---

## Contributing

Contributions to either module are welcome. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Describe your change"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request with a clear description of what was changed and why.

Please ensure code is commented where appropriate and that any new dependencies are documented.

---

## License

This project is open-source. If you use or build upon this work in academic or applied research, attribution to the original author is appreciated.

---

**Author**: [tejaswantt](https://github.com/tejaswantt)  
**Repository**: [a-framework-for-instagram-reach-analysis-using-ml](https://github.com/tejaswantt/a-framework-for-instagram-reach-analysis-using-ml)
