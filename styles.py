import streamlit as st


def apply_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 4px;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #555;
            margin-top: 0;
        }

        .footer {
            text-align: center;
            color: #aaa;
            font-size: 0.85rem;
            padding: 8px 0;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed #bbb;
            border-radius: 10px;
            padding: 10px;
        }

        hr {
            border-color: #eee;
        }
    </style>
    """, unsafe_allow_html=True)
