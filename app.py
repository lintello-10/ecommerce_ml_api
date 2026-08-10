import requests
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="E-Commerce Purchase Predictor (API Powered)",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS STYLING FOR A PREMIUM LOOK
# ==========================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f17;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: #0b0f17;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .metric-card {
        background-color: #111622;
        border: 1px solid #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown(
    "<h1 style='text-align: center; color: #ffffff;'>🛒 E-Commerce Purchase Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>Real-time user purchase intent prediction powered by a <strong>FastAPI MLOps backend</strong> deployed on Render.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ==========================================
# SIDEBAR - API STATUS & CONFIGURATION
# ==========================================
st.sidebar.header("🔌 Backend Connection")
# Replace with your actual Render API endpoint URL
API_BASE_URL = st.sidebar.text_input(
    "FastAPI Endpoint URL",
    value="https://ecommerce-ml-api-gqub.onrender.com",
)

# Check API Health status on load
health_url = f"{API_BASE_URL}/health"
try:
  health_response = requests.get(health_url, timeout=5)
  if health_response.status_code == 200:
    st.sidebar.success("🟢 API Status: Online & Ready")
  else:
    st.sidebar.warning("🟡 API Status: Responding with errors")
except requests.exceptions.RequestException:
  st.sidebar.error("🔴 API Status: Offline / Spinning up...")
  st.sidebar.info(
      "Note: Free Render instances take ~50s to wake up on the first request!"
  )

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developed by Dramé Bourama")
st.sidebar.markdown(
    "L3 Math-Info Student | Aspiring Data Scientist & MLOps Engineer"
)

# ==========================================
# MAIN FORM - USER INPUTS
# ==========================================
st.subheader("📊 Session Behavior & Demographics")
st.markdown(
    "Enter the user interaction metrics and demographic details below:"
)

with st.form("prediction_form"):
  col1, col2 = st.columns(2)

  with col1:
    # Behavioral features (funnel milestones)
    count_view_item = st.number_input(
        "View Item Count",
        min_value=0,
        value=3,
        step=1,
        help="Number of times the user viewed an item.",
    )
    count_add_to_cart = st.number_input(
        "Add to Cart Count",
        min_value=0,
        value=1,
        step=1,
        help="Number of times the user added items to cart.",
    )
    count_begin_checkout = st.number_input(
        "Begin Checkout Count",
        min_value=0,
        value=0,
        step=1,
        help="Number of times the user initiated checkout.",
    )

  with col2:
    # Contextual features
    device_category = st.selectbox(
        "Device Category",
        options=["desktop", "mobile", "tablet"],
        help="Type of device used during the session.",
    )
    traffic_medium = st.selectbox(
        "Traffic Medium",
        options=["organic", "cpc", "referral", "none", "affiliate"],
        help="Channel source medium for the visit.",
    )
    country = st.text_input(
        "Geographical Country",
        value="United States",
        help="Country of origin of the user session.",
    )

  # Submit button for prediction
  submit_button = st.form_submit_button(
      label="⚡ Predict Purchase Intent via API"
  )

# ==========================================
# API REQUEST HANDLING & RESULTS DISPLAY
# ==========================================
if submit_button:
  # Construct the payload matching the FastAPI Pydantic schema
  payload = {
      "count_view_item": int(count_view_item),
      "count_add_to_cart": int(count_add_to_cart),
      "count_begin_checkout": int(count_begin_checkout),
      "device_category": str(device_category),
      "traffic_medium": str(traffic_medium),
      "country": str(country),
  }

  predict_url = f"{API_BASE_URL}/predict"

  # Show a spinner while communicating with the cloud API
  with st.spinner(
      "Sending request to FastAPI backend on Render (may take ~50s if waking"
      " up)..."
  ):
    try:
      response = requests.post(predict_url, json=payload, timeout=60)

      if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction")
        probability = result.get("probability", None)

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        # Display success box based on model output
        if prediction == 1:
          st.success(
              "🎉 **High Purchase Intent Detected!** This user is very likely to"
              " complete a purchase."
          )
        else:
          st.info(
              "🔍 **Low Purchase Intent.** This user is unlikely to convert"
              " during this session."
          )

        # Show probability if available in response
        if probability is not None:
          st.metric(
              label="Conversion Probability Score", value=f"{probability:.2%}"
          )

      else:
        st.error(
            f"❌ API Error (Status {response.status_code}):"
            f" {response.text}"
        )

    except requests.exceptions.Timeout:
      st.error(
          "⏳ Request timed out. The server on Render took too long to respond."
          " Please try again!"
      )
    except requests.exceptions.RequestException as e:
      st.error(
          f"❌ Failed to connect to the API endpoint. Technical error: {e}"
      )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Built"
    " with Streamlit & FastAPI | MLOps Architecture</p>",
    unsafe_allow_html=True,
)