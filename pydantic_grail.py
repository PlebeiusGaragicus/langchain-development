import json
from time import sleep
from enum import Enum

# pip install pydantic==1.9
from pydantic import BaseModel, Field, ValidationError

import streamlit as st
import streamlit_pydantic as sp
from stqdm import stqdm

# Define an Enum for dropdown options
class DropdownOptions(Enum):
    option1 = "Option 1"
    option2 = "Option 2"
    option3 = "Option 3"



class ExampleModel(BaseModel):
    refine_node_prompt: str = Field(
        default="You are an expert...",
        # ...,
        title="Prompt for `refine` node LLM",
        description="Prompt for the refine node"
    )
    refine_llm_temperature: float = Field(
        # ...,
        default=0.45,
        title="Temperature for `refine` node LLM",
        description="This is the temperature for the llm used in the refine node",
        ge=0.0,
        le=1.0
    )
    number_of_iterations: int = Field(
        # ...,
        default=2,
        title="Number of refinement iterations",
        description="Number of refinement steps",
        ge=1,
        le=10
    )
    skip_refine_node: bool = Field(
        # ...,
        # default=False, # NOTE: This must default to False or else there's a bug!!! (and it will ALWAYS default to True in the UI)
        # default=True,
        title="Skip the `refine` node?",
        description="Skip `refine` node?"
    )
    # dropdown_selection: DropdownOptions = Field(
    #     default="option1",
    #     title="Choose an option",
    #     description="Select your option from the dropdown"
    # )



# try to open settings.json
try:
    with open("settings.json", "r") as f:
        settings = json.load(f)
        print()
        print(settings)

    st.json(settings)
    the_model = ExampleModel(**settings) if settings else None
    st.json(the_model.json())
except (FileNotFoundError, json.JSONDecodeError, ValidationError):
    st.error("No settings file found")
    # settings = ExampleModel()
    the_model = ExampleModel()

print("Model after loading:", the_model.json())  # Debug print


if data := sp.pydantic_form(key="my_form", model=the_model, submit_label="Save Settings", ignore_empty_values=True):
# if data := sp.pydantic_input(key="my_form", model=the_model):
    st.json(data.json())

    # save to a file
    with open("settings.json", "w") as f:
        f.write(data.json())

    
    # time.sleep(1)
    # for _ in stqdm(range(3)):
    for _ in stqdm(range(60)):
        sleep(0.01)

    st.rerun() # nasty fucking bug..