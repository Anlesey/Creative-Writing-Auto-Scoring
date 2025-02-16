import streamlit as st

with open('Readme.md', encoding='utf-8') as f:
    st.markdown(f.read())


if st.button('来试试看吧', use_container_width = True):
    st.switch_page("pages/1 singleton.py")
