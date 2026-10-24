
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Job Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# Find the main Project 3 folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
data_path = PROJECT_DIR / "data" / "engineered_job_placement_data.csv"
df = pd.read_csv(data_path)

# Dashboard title
st.title("🎯 Job Acceptance Prediction System")
st.caption(
    "HR placement analytics and machine-learning predictions"
)
# Summary metrics
total_candidates = len(df)
placed_candidates = (df["status"] == "Placed").sum()
not_placed_candidates = (df["status"] == "Not Placed").sum()
placement_rate = placed_candidates / total_candidates * 100

avg_interview_score = df["average_interview_score"].mean()
avg_skills_match = df["skills_match_percentage"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Candidates", f"{total_candidates:,}")
col2.metric("Placement Rate", f"{placement_rate:.2f}%")
col3.metric("Avg Interview Score", f"{avg_interview_score:.2f}")
col4.metric("Avg Skills Match", f"{avg_skills_match:.2f}%")

#st.divider()

# Navigation tabs
overview_tab, candidate_tab, model_tab, prediction_tab = st.tabs([
    "📊 Overview",
    "🎓 Candidate Analysis",
    "🤖 Model Performance",
    "🔮 Predict a Candidate"
])

# Overview Tab
with overview_tab:
    col1, col2 = st.columns(2, gap="medium")

    # Chart 1: Placement Status
    with col1:
        with st.container(border=True):
            st.subheader("Placement Status", anchor=False)

            status_counts = df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig = px.bar(
                status_counts,
                x="Status",
                y="Count"
            )

            fig.update_traces(width=0.4)
            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Placement Rate by Experience
    with col2:
        with st.container(border=True):
            st.subheader("Placement Rate by Experience", anchor=False)

            experience_analysis = df.groupby(
                "experience_category"
            )["status"].apply(
                lambda x: (x == "Placed").mean() * 100
            ).reset_index()

            experience_analysis.columns = [
                "Experience Category",
                "Placement Rate"
            ]

            fig = px.bar(
                experience_analysis,
                x="Experience Category",
                y="Placement Rate",
                labels={"Placement Rate": "Placement Rate (%)"}
            )

            fig.update_traces(width=0.4,marker_color="#14B8A6")
            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

# Candidate Analysis

with candidate_tab:
    # Row 1: Two charts
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.subheader("Academic Scores vs Placement")

            academic_analysis = df.groupby("status")[
                [
                    "ssc_percentage",
                    "hsc_percentage",
                    "degree_percentage"
                ]
            ].mean().reset_index()

            academic_analysis = academic_analysis.melt(
                id_vars="status",
                var_name="Academic Score",
                value_name="Average Percentage"
            )

            fig = px.bar(
                academic_analysis,
                x="status",
                y="Average Percentage",
                color="Academic Score",
                barmode="group",
                color_discrete_sequence=[
                    "#6366F1", "#14B8A6", "#F59E0B"
                ]
            )

            fig.update_traces(width=0.22)
            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Placement Rate by Company Tier")

            tier_analysis = df.groupby("company_tier")[
                "status"
            ].apply(
                lambda x: (x == "Placed").mean() * 100
            ).reset_index(name="Placement Rate")

            fig = px.bar(
                tier_analysis,
                x="company_tier",
                y="Placement Rate",
                color_discrete_sequence=["#8B5CF6"]
            )

            fig.update_traces(width=0.4)
            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Two charts
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        with st.container(border=True):
            st.subheader("Certifications vs Placement")

            certification_analysis = df.groupby(
                "certifications_count"
            )["status"].apply(
                lambda x: (x == "Placed").mean() * 100
            ).reset_index(name="Placement Rate")

            fig = px.bar(
                certification_analysis,
                x="certifications_count",
                y="Placement Rate",
                color_discrete_sequence=["#F59E0B"]
            )

            fig.update_traces(width=0.4)
            fig.update_layout(
                height=350,
                margin=dict(t=80),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.15,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.subheader("Interview Performance vs Placement")

            interview_analysis = df.groupby(
                "interview_performance_category"
            )["status"].apply(
                lambda x: (x == "Placed").mean() * 100
            ).reset_index(name="Placement Rate")

            fig = px.bar(
                interview_analysis,
                x="interview_performance_category",
                y="Placement Rate",
                color_discrete_sequence=["#EC4899"]
            )

            fig.update_traces(width=0.4)
            fig.update_layout(height=350)

            st.plotly_chart(fig, use_container_width=True)

# Machine Learning
with model_tab:
    st.subheader("🤖 Machine Learning Model Comparison")

    comparison_path = PROJECT_DIR / "data" / "model_comparison.csv"

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)

        # Display model comparison table
        st.write("Compare the performance of all five models.")

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1_Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Compare model metrics visually
        st.subheader("Performance Metrics by Model")

        chart_df = comparison_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1_Score"]
        ]

        st.bar_chart(chart_df)

        # Identify the model with the highest F1 score
        best_model_row = comparison_df.loc[
            comparison_df["F1_Score"].idxmax()
        ]

        st.success(
            f"Best model by F1-score: {best_model_row['Model']} "
            f"({best_model_row['F1_Score']:.2%})"
        )

    else:
        st.error(
            "Model comparison file not found. "
            "Please check data/model_comparison.csv."
        )




with prediction_tab:
    st.subheader("🔮 Predict a Candidate")
    st.caption(
        "Enter candidate details to predict the placement outcome."
    )

    import joblib

    model_path = PROJECT_DIR / "models" / "placement_models.pkl"

    if model_path.exists():
        trained_models = joblib.load(model_path)
        best_model = trained_models["Random Forest"]

        candidate = {}

        def render_fields(fields, section_title):
            st.markdown(f"#### {section_title}")

            # Display four fields per row
            for start in range(0, len(fields), 4):
                field_row = fields[start:start + 4]
                cols = st.columns(4)

                for col, field in zip(cols, field_row):
                    with col:
                        if field in numeric_fields:
                            candidate[field] = st.number_input(
                                field.replace("_", " ").title(),
                                min_value=float(df[field].min()),
                                max_value=float(df[field].max()),
                                value=float(df[field].median()),
                                key=f"candidate_{field}",
                            )
                        else:
                            options = sorted(
                                df[field].dropna().unique().tolist()
                            )
                            mode = df[field].mode()
                            default_index = (
                                options.index(mode.iloc[0])
                                if not mode.empty
                                and mode.iloc[0] in options
                                else 0
                            )

                            candidate[field] = st.selectbox(
                                field.replace("_", " ").title(),
                                options,
                                index=default_index,
                                key=f"candidate_{field}",
                            )

        # Original input features
        numeric_fields = [
            "age_years",
            "ssc_percentage",
            "hsc_percentage",
            "degree_percentage",
            "technical_score",
            "aptitude_score",
            "communication_score",
            "skills_match_percentage",
            "certifications_count",
            "years_of_experience",
            "previous_ctc_lpa",
            "expected_ctc_lpa",
            "notice_period_days",
            "employment_gap_months",
        ]

        categorical_fields = [
            "gender",
            "degree_specialization",
            "internship_experience",
            "career_switch_willingness",
            "relevant_experience",
            "company_tier",
            "job_role_match",
            "competition_level",
            "bond_requirement",
            "layoff_history",
            "relocation_willingness",
        ]

        with st.form("candidate_prediction_form"):

            render_fields([
                "age_years",
                "gender",
                "degree_specialization",
                "ssc_percentage",
                "hsc_percentage",
                "degree_percentage",
                "skills_match_percentage",
                "certifications_count",
            ], "🎓 Academics & Skills")

            st.divider()

            render_fields([
                "technical_score",
                "aptitude_score",
                "communication_score",
                "years_of_experience",
                "internship_experience",
                "relevant_experience",
                "career_switch_willingness",
            ], "🧠 Tests & Experience")

            st.divider()

            render_fields([
                "company_tier",
                "job_role_match",
                "competition_level",
                "bond_requirement",
                "previous_ctc_lpa",
                "expected_ctc_lpa",
                "notice_period_days",
                "employment_gap_months",
                "relocation_willingness",
                "layoff_history",
            ], "💼 Offer & Expectations")

            predict_clicked = st.form_submit_button(
                "🔮 Predict Placement",
                type="primary",
                use_container_width=True,
            )

        if predict_clicked:
            # Calculate engineered features as during training
            experience = candidate["years_of_experience"]
            candidate["experience_category"] = (
                "Fresher" if experience == 0
                else "Junior" if experience <= 2
                else "Senior"
            )

            degree_score = candidate["degree_percentage"]
            candidate["academic_performance_band"] = (
                "Low" if degree_score < 60
                else "Medium" if degree_score < 75
                else "High"
            )

            skills_score = candidate["skills_match_percentage"]
            candidate["skills_match_level"] = (
                "Low" if skills_score < 60
                else "Medium" if skills_score < 80
                else "High"
            )

            interview_score = (
                candidate["technical_score"]
                + candidate["aptitude_score"]
                + candidate["communication_score"]
            ) / 3

            candidate["average_interview_score"] = interview_score

            candidate["interview_performance_category"] = (
                "Low" if interview_score < 60
                else "Medium" if interview_score < 75
                else "High"
            )

            feature_names = list(best_model.feature_names_in_)
            candidate_df = pd.DataFrame([candidate])[feature_names]

            prediction = best_model.predict(candidate_df)[0]
            classes = list(best_model.classes_)
            placed_index = classes.index("Placed")

            probability = (
                best_model.predict_proba(candidate_df)[0, placed_index]
                * 100
            )

            st.divider()
            st.subheader("📋 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric("Predicted Status", prediction)
            result_col2.metric(
                "Placement Probability",
                f"{probability:.2f}%",
            )

            st.progress(int(round(probability)))

            if prediction == "Placed":
                st.success(
                    "The model predicts that this candidate will be placed."
                )
            else:
                st.warning(
                    "The model predicts that this candidate will not be placed."
                )

            st.caption(
                "This is a model estimate, not a guarantee of placement."
            )

    else:
        st.error(f"Model file not found: {model_path}")


