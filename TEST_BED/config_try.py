def create_slider(name, min_value, max_value, default=None, step=None):
    return {
        "name": name,
        "type": "float",
        "widget": "slider",
        "min": min_value,
        "max": max_value,
        "default": default if default is not None else min_value,
        "step": step
    }

def create_selectbox(name, options, default=None):
    return {
        "name": name,
        "type": "string",
        "widget": "selectbox",
        "options": options,
        "default": default if default else options[0]
    }

def create_checkbox(name, default=False):
    return {
        "name": name,
        "type": "bool",
        "widget": "checkbox",
        "default": default
    }

def create_text_area(name, default=''):
    return {
        "name": name,
        "type": "string",
        "widget": "text_area",
        "default": default
    }



import streamlit as st

# Directly define the configuration as a dictionary
something = {
    "graph_name": "Example Graph",
    "widgets": [
        create_text_area("wooske", default="back to back"),
        create_slider("threshold", 0.0, 1.0, default=0.5),
        create_selectbox("operation", ["add", "subtract", "multiply", "divide"]),
        create_checkbox("enable_feature"),
        create_text_area("remarks", default="Initial remark...")
    ]
}


# st.title(config["graph_name"])
# graph_hyperparameters = {}

def build_interface(config):
    if "graph_hyperparameters" not in st.session_state:
        st.session_state.graph_hyperparameters = {}

    st.header(":orange[Θ] Hyperparameters", divider="rainbow")
    with st.container(border=True):

        for widget in config["widgets"]:
            if widget["widget"] == "slider":
                st.session_state.graph_hyperparameters[widget["name"]] = st.slider(
                    label=widget["name"],
                    min_value=widget["min"],
                    max_value=widget["max"],
                    value=widget["default"]
                )
            elif widget["widget"] == "selectbox":
                st.session_state.graph_hyperparameters[widget["name"]] = st.selectbox(
                    label=widget["name"],
                    options=widget["options"]
                )
            elif widget["widget"] == "checkbox":
                st.session_state.graph_hyperparameters[widget["name"]] = st.checkbox(
                    label=widget["name"],
                    value=widget["default"]
                )
            elif widget["widget"] == "text_area":
                st.session_state.graph_hyperparameters[widget["name"]] = st.text_area(
                    label=widget["name"],
                    value=widget["default"]
                )

build_interface(something)

if st.button('Run Graph'):
    # Use `graph_hyperparameters` to build the config and invoke your LangGraph Runnable
    st.write("Running with config:", st.session_state.graph_hyperparameters)


with st.sidebar:
    st.write(st.session_state)
