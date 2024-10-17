import re
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

class GraphGenerator:
    def __init__(self, chain, csv_string):
        self.chain = chain
        self.csv_string = csv_string

    def get_fig_from_code(self, code):
        local_variables = {}
        exec(code, {}, local_variables)
        fig = local_variables.get('fig', None)

        if fig is not None:
            for trace in fig.data:
                if 'x' in trace:
                    trace['x'] = pd.Series(trace['x']).astype(str).tolist()
                if 'y' in trace:
                    trace['y'] = pd.Series(trace['y']).astype(str).tolist()

        return fig

    def render(self):
        st.subheader("Generate a Graph from CSV Data")
        user_input = st.text_area("Enter your request for a graph:", height=50)

        if st.button("Submit Graph Request"):
            response = self.chain.invoke(
                {
                    'messages': [HumanMessage(content=user_input)],
                    'data': self.csv_string
                }
            )
            result_output = response.content
            code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', result_output, re.DOTALL)

            if code_block_match:
                code_block = code_block_match.group(1).strip()
                cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
                fig = self.get_fig_from_code(cleaned_code)
                st.plotly_chart(fig)