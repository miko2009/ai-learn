from typing import Annotated

from numpy import ones
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.llms import Tongyi
import operator
import dotenv
dotenv.load_dotenv()
llm = Tongyi(
    temperature=0.1,
)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    score: Annotated[int, operator.add]

graph_builder = StateGraph(State)


def prepareEgg(state):
    print("start ---\n")
    print(state)
    return state

def scramed(state):
    print("node--scramed")

def oneSlide(state):
    print('one-slide')
def hardLine(state):
    print('hard----line')

def checkEggType(state):
    if state['score'] > 70:
        return "scramed"
    if state['score'] < 30:
        return "oneSlided"
    else:
        return "hardbolled"
def end(state):
    print('-----end-----')

graph_builder.add_node("prepareEgg", prepareEgg)
graph_builder.add_node("scramed", scramed)
graph_builder.add_node("oneSlided", oneSlide)
graph_builder.add_node("hardbolled", hardLine)
graph_builder.add_node("end", end)

graph_builder.set_entry_point("prepareEgg")
graph_builder.add_conditional_edges("prepareEgg", checkEggType, {"scramed": "scramed", "oneSlided": "oneSlided",  "hardbolled": "hardbolled" })
graph_builder.add_edge("scramed", "end")
graph_builder.add_edge("oneSlided", "end")
graph_builder.add_edge("hardbolled", "end")
graph_builder.set_finish_point("end")
graph = graph_builder.compile()
graph.invoke({"score": 50})