import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_distribution(df, selected_parameter):
    """Plots the distribution of a selected parameter."""
    
    try:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_parameter], kde=True, ax=ax)
        ax.set_title(f'Distribution of {selected_parameter}')
        ax.set_xlabel(selected_parameter)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    except Exception as e:
        st.write(f"An error occurred while plotting: {e}")