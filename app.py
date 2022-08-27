import streamlit as st
import awesome_streamlit as ast

st.set_page_config(page_title="CPR", initial_sidebar_state="expanded", layout="wide")

import app_en
import app_fr


st.markdown(f"""
    <style>
    .appview-container .main .block-container{{
    padding-top: {0}em;
    padding-left: {3}em;
    padding-right: {3}em;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 120px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            /* Position the tooltip */
            position: absolute;
            z-index: 1;
            top: 100%;
            left: 50%;
            margin-left: -60px;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
        }}
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
    font-size: 16px;
    font-weight: 600;
    }
    </style>""", unsafe_allow_html=True)


# hide the hamburger button top-right of the screen
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
    </style>""", unsafe_allow_html=True)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.markdown("""
    <style>
        div.block-container {padding-top:0rem;}
    </style>""", unsafe_allow_html=True)

PAGES = {"Français": app_fr, "English": app_en}

def main():
    st.sidebar.markdown("# Langue / Language")
    selection = st.sidebar.radio(" ", list(PAGES.keys()))
    page = PAGES[selection]

    load_msg = {"English": "Please wait...",
                "Français": "Veuillez patienter..."}
    with st.spinner(load_msg[selection]):
        ast.shared.components.write_page(page)

if __name__ == "__main__":
	main()