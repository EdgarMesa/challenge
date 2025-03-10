import os
import sys
import sqlite3
from pathlib import Path
from pprint import pprint


from tqdm import tqdm
from pydantic import BaseModel, Field
import google.generativeai as genai

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    RemoveMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)
from langgraph.prebuilt import ToolNode


root_dir = Path(os.getcwd()).parent.parent

from src.d01_data.pydantic_classes import Resumen
from src.d01_data.prompts import CONVERSATION_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT
from src.d03_modeling.modeling import consult_legal_database
 
memories_path = root_dir / 'data' / '03_memories'

db_path = memories_path / 'short_term_memory.db'
conn = sqlite3.connect(str(db_path), check_same_thread=False)
short_term_memory = SqliteSaver(conn)

tools = [consult_legal_database]
tool_node = ToolNode(tools)

conversation_llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL'),
    temperature=0.6,
    top_p=0.6,
    max_tokens=500,
    timeout=None,
    max_retries=2).bind_tools(tools)

sumarization_llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL'),
    temperature=0,
    top_p=0.6,
    max_tokens=500,
    timeout=None,
    max_retries=2)
sumarization_llm = sumarization_llm.with_structured_output(Resumen, include_raw=False)

token_counter = genai.GenerativeModel(os.getenv('GEMINI_MODEL'))

class InnerState(MessagesState):
    resumen: str = Field(title='Resumen', description='Resumen de la conversación')
class OutputState(MessagesState):
    pass

def conversation(state: InnerState) -> OutputState:
    """
    Handles a conversation by constructing a message history and invoking a language model.
    """
    
    # Retrieve the summary of the conversation if it exists; otherwise, use an empty string
    summary = state.get('resumen', '')
    if summary:
        system_message = summary
        
        # Construct the message history by prepending the system message to the existing messages
        messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT.format(system_message))] \
                    + state['messages']
    else:
        messages = state['messages']
    response = conversation_llm.invoke(messages)
    
    return {'messages': response}


def summarize_conversation(state: InnerState) -> InnerState:
    """
    Summarizes the conversation based on the given state.
    """

    prev_summary = state.get('resumen', 'No resumen previo')

    # Construct the message history with the summarization prompt
    messages = [SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT.format(prev_summary))] + state['messages']
    
    response = sumarization_llm.invoke(messages)
    # Remove older messages to manage token usage, keeping the most recent 6 messages
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-6]]

    return {'resumen': response.resumen, 'messages': delete_messages}


def should_continue(state: InnerState):
    """Return the next node to execute."""
    
    messages = state['messages']
    
     
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    tokens = token_counter.count_tokens('\n'.join([m.content for m in messages])).total_tokens
   
    if tokens > 5_000:
        return 'summarize_conversation'
    
    return END


workflow = StateGraph(InnerState, input=OutputState, output=OutputState)
workflow.add_node('conversation', conversation)
workflow.add_node(summarize_conversation)
workflow.add_node('tools', tool_node)


workflow.add_edge(START, 'conversation')
workflow.add_conditional_edges('conversation', should_continue)
workflow.add_edge('summarize_conversation', END)
workflow.add_edge('tools', 'conversation')

graph = workflow.compile(checkpointer=short_term_memory)


__all__ = ['graph']