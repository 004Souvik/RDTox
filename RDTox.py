import streamlit as st
import pandas as pd
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lib.leverage import leverage_calculator
from PIL import Image

st.set_page_config(page_title="RDTox", layout="centered")

# --- Utility functions ---
@st.cache_data
def load_train_data():
    # load training files (paths relative to the app location)
    l408 = pd.read_excel("lib/oecd408/Train.xlsx", index_col=0)
    l407_422 = pd.read_excel("lib/oecd407&422/Train.xlsx", index_col=0)
    return l408, l407_422


def stand(df1, df2):
    avg = df1.mean()
    stdev = df1.std()
    std_df1 = (df1 - avg) / stdev
    std_df2 = (df2 - avg) / stdev
    return std_df1, std_df2


def to_excel_bytes(dfs_dict):
    # write multiple dataframes to an in-memory Excel and return bytes
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    return output.getvalue()


# --- Sidebar ---
st.sidebar.title("RDTox - Controls")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"]) 
show_preview = st.sidebar.checkbox("Show uploaded dataset preview", value=True)
run_button = st.sidebar.button("Run prediction")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

try:
    with open("Sample.xlsx", "rb") as f:
        sample_bytes = f.read()

    st.sidebar.download_button(
        label="📥 Sample File",
        data=sample_bytes,
        file_name="Sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except FileNotFoundError:
    st.sidebar.warning("Sample.xlsx not found in root directory.")

# optional: show background / logo if exists
try:
    logo = Image.open("icon/bg.png")
    st.image(logo, use_column_width=True)
except Exception:
    pass

st.title("RDTox")
st.write("Upload an Excel file (first column as index). The app will run two models (OECD TG 408 LOAEL and OECD TG 407&422 LOAEL) and return predicted classes + AD Status.")

l408, l407_422 = load_train_data()

if uploaded_file is not None:
    try:
        target = pd.read_excel(uploaded_file, index_col=0)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

    if show_preview:
        st.subheader("Uploaded dataset preview")
        st.dataframe(target.head())

    if run_button:
        with st.spinner("Running predictions..."):
            try:
                # prepare train/test descriptors
                xtr_l408 = l408.iloc[:, :-1]
                ytr_l408 = l408.iloc[:, -1]
                xtr_l407_422 = l407_422.iloc[:, :-1]
                ytr_l407_422 = l407_422.iloc[:, -1]

                # prepare test sets using train columns
                test_408 = target.reindex(columns=xtr_l408.columns).copy()
                test_407_422 = target.reindex(columns=xtr_l407_422.columns).copy()

                # standardize
                std_xtr_l408, std_test_408 = stand(df1=xtr_l408, df2=test_408)
                std_xtr_l407_422, std_test_407_422 = stand(df1=xtr_l407_422, df2=test_407_422)

                # fit models
                cls2 = LogisticRegression(random_state=0, max_iter=1000).fit(std_xtr_l408.fillna(0), ytr_l408)
                cls4 = GaussianNB().fit(std_xtr_l407_422.fillna(0), ytr_l407_422)

                # predictions OECD 408
                d2_pred = pd.DataFrame(cls2.predict(std_test_408.fillna(0)), columns=["Predicted Class"], index=target.index)
                d2_pred["Predicted Class"] = d2_pred["Predicted Class"].map({1: "More Toxic", 0: "Less Toxic"})
                tr_lev, te_lev, h_star = leverage_calculator(data1=xtr_l408, data2=test_408)
                # ensure alignment
                d2_pred["AD Status"] = te_lev.reindex(d2_pred.index)["AD Status"].values

                # predictions OECD 407&422
                d4_pred = pd.DataFrame(cls4.predict(std_test_407_422.fillna(0)), columns=["Predicted Class"], index=target.index)
                d4_pred["Predicted Class"] = d4_pred["Predicted Class"].map({1: "More Toxic", 0: "Less Toxic"})
                tr_lev1, te_lev1, h_star1 = leverage_calculator(data1=xtr_l407_422, data2=test_407_422)
                d4_pred["AD Status"] = te_lev1.reindex(d4_pred.index)["AD Status"].values

                # show results
                st.success("Prediction completed")
                st.subheader("OECD TG 408 Results")
                st.dataframe(d2_pred)

                st.subheader("OECD TG 407 & 422 Results")
                st.dataframe(d4_pred)

                # prepare excel for download
                excel_bytes = to_excel_bytes({"OECD 408": d2_pred, "OECD 407&422": d4_pred})
                st.download_button("Download results as Excel", data=excel_bytes, file_name="Results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Upload an .xlsx file in the sidebar to get started.")

# small footer
st.markdown("---")






