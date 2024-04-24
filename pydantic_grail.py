from time import sleep
import json
from stqdm import stqdm

# pip install pydantic==1.9
from pydantic import BaseModel, Field

import streamlit as st
import streamlit_pydantic as sp


class ExampleModel(BaseModel):
    refine_node_prompt: str = Field(
        # alias="Prompt for `refine` node LLM",
        default="You are an expert...",
        description="Prompt for the refine node"
    )
    refine_llm_temperature: float = Field(
        # alias="Temperature for `refine` node LLM",
        default=0.45,
        description="This is the temperature for the llm used in the refine node",
        ge=0.0,
        le=1.0
    )
    number_of_iterations: int = Field(
        # alias="Number of refinement iterations",
        default=2,
        description="Number of refinement steps",
        ge=1,
        le=10
    )
    skip_refine_node: bool = Field(
        # alias="Skip the `refine` node?",
        default=True,
        description="Skip `refine` node?"
    )



# try to open settings.json
try:
    with open("settings.json", "r") as f:
        settings = json.load(f)
        print()
        print(settings)

    the_model = ExampleModel(**settings) if settings else None
    st.json(the_model.json())
except FileNotFoundError:
    st.error("No settings file found")
    settings = None
    the_model = ExampleModel()


if data := sp.pydantic_form(key="my_form", model=the_model, submit_label="Save Settings"):
    st.json(data.json())

    # save to a file
    with open("settings.json", "w") as f:
        f.write(data.json())

    
    # time.sleep(1)
    # for _ in stqdm(range(3)):
    for _ in stqdm(range(60)):
        sleep(0.01)

    st.rerun() # nasty fucking bug..