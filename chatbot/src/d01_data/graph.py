import os
import sys
import sqlite3
import uuid
from pathlib import Path
from pprint import pprint

from tqdm import tqdm
from pydantic import BaseModel, Field

import google.generativeai as genai

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
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
from langchain_core.runnables.config import RunnableConfig  

root_dir = Path(os.getcwd()).parent.parent
memories_path = root_dir / 'data' / '03_memories'

from src.d01_data.pydantic_classes import Resumen, PerfilUsuario
from src.d01_data.prompts import CONVERSATION_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT, EXTRACT_LONG_MEMORY_SYSTEM_PROMPT
from src.d03_modeling.modeling import consult_legal_database
from src.d01_data.data import create_usuarios_table, get_user_info, upsert_user
from src.d00_utils.utils import update_dict_preserve

db_path = memories_path / 'short_term_memory.db'
conn = sqlite3.connect(str(db_path), check_same_thread=False)
short_term_memory = SqliteSaver(conn)

long_term_db_path = memories_path / 'long_term_memory.db'
long_term_db_conn = sqlite3.connect(str(long_term_db_path), check_same_thread=False)
create_usuarios_table(long_term_db_conn)
 
tools = [consult_legal_database]
# tool_node = ToolNode(tools)

conversation_llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL'),
    temperature=0.6,
    top_p=0.6,
    max_tokens=500,
    timeout=None,
    max_retries=2
).bind_tools(tools)

sumarization_llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL'),
    temperature=0,
    top_p=0.6,
    max_tokens=500,
    timeout=None,
    max_retries=2
)
sumarization_llm = sumarization_llm.with_structured_output(Resumen, include_raw=False)

extract_user_info_llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL'),
    temperature=0,
    top_p=0.7,
    max_tokens=100,
    timeout=None,
    max_retries=2
)
extract_user_info_llm = extract_user_info_llm.with_structured_output(PerfilUsuario, include_raw=False)


token_counter = genai.GenerativeModel(os.getenv('GEMINI_MODEL'))

tools_by_name = {tool.name: tool for tool in tools}

class InnerState(MessagesState):
    resumen: str = Field(title='Resumen', description='Resumen de la conversación')
    perfil_usuario: dict
class OutputState(MessagesState):
    pass

def conversation(state: InnerState, config: RunnableConfig) -> InnerState:
    """
    Handles a conversation by constructing a message history and invoking a language model.
    """
    
    user_id = config['configurable']['user_id']
    user_info = get_user_info(long_term_db_conn, user_id)
    if not user_info:
        user_info = {}
    
    summary = state.get('resumen', '')
    if summary:
        system_message = summary
        messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT.format(system_message, user_info))] + state['messages']
    else:
        messages = state['messages']
    response = conversation_llm.invoke(messages)
    return {'messages': response, 'perfil_usuario': user_info}



def consult_database(state: InnerState) -> OutputState:
    """
    Consult the legal database based on the provided state.
    """
    messages = state['messages']
    
    last_message = messages[-1]
    
    outputs = []
    
    # Iterate over all tool calls in the last message.
    for tool_call in last_message.tool_calls:
        tool_call_name = tool_call['name']
        
        # Validate that the tool call name is 'consult_legal_database'.
        if tool_call_name != 'consult_legal_database':
            raise Exception(f'Tool call name "{tool_call_name}" does not match "consult_legal_database"')
        
        tool_result = tools_by_name[tool_call_name].invoke(tool_call['args'])
        
        # Append the result wrapped in a ToolMessage object to the outputs list.
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call_name,
                tool_call_id=tool_call['id'],
            )
        )
    
    return {'messages': outputs}

def summarize_conversation(state: InnerState) -> InnerState:
    """
    Summarizes the conversation based on the given state.
    """
    prev_summary = state.get('resumen', 'No resumen previo')
    messages = [SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT.format(prev_summary))] + state['messages']
    response = sumarization_llm.invoke(messages)
    # Remove older messages to manage token usage, keeping the most recent 6 messages
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-6]]
    return {'resumen': response.resumen, 'messages': delete_messages}



def update_long_term_memory(state: InnerState, config: RunnableConfig) -> OutputState:
    """
    Checks if the conversation contains long term profile information.
    If so, it tries to extract the details using an LLM model and updates the
    long term memory SQLite database with the user_id from config.
    """
    
    last_user_message = ''
    
    for mess in state['messages'][::-1]:
        if isinstance(mess, HumanMessage):
            last_user_message = mess
            break
            
    if last_user_message:
        user_id = config['configurable']['user_id']
        user_info = state['perfil_usuario']
        user_info_str = '' if not user_info else str(user_info)
        messages = [SystemMessage(content=EXTRACT_LONG_MEMORY_SYSTEM_PROMPT.format(user_info_str))] + [last_user_message]
        
        response = extract_user_info_llm.invoke(messages)
        
        id_tool_call = str(uuid.uuid4())
        if response.actualizar == 'si':
            
            nombre = response.nombre if response.nombre else None
            profesion = response.profesion if response.profesion else None
            gmail = response.gmail if response.gmail else None
            
            new_info = {'nombre':nombre, 'profesion':profesion, 'gmail':gmail}
            new_info = update_dict_preserve(user_info, new_info)
            upsert_user(long_term_db_conn, user_id=user_id, user_data=new_info)
            
            output = ToolMessage(content=f'Se han actualizado la información personal del usuario {user_id}:\nNuevos valores:{new_info}',
                        tool_call_id=id_tool_call)
        else:
            output = ToolMessage(content=f'No hay nuevos datos personales que actualizar para el usuario {user_id}',
                        tool_call_id=id_tool_call)
            
    return {'messages': [output]}
    

def should_continue(state: InnerState):
    """Return the next node to execute."""
    messages = state['messages']
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        tool_call_name = last_message.tool_calls[0]['name']
        if tool_call_name == 'consult_legal_database':
            return 'consult_database'
        
    tokens = token_counter.count_tokens('\n'.join([m.content for m in messages])).total_tokens
    if tokens > 5_000:
        return 'summarize_conversation'
    return 'update_memory'



workflow = StateGraph(InnerState, input=OutputState, output=OutputState)


workflow.add_node('conversation', conversation)
workflow.add_node('summarize_conversation', summarize_conversation)
workflow.add_node('consult_database', consult_database)
workflow.add_node('update_memory', update_long_term_memory)

workflow.add_edge(START, 'conversation')
workflow.add_conditional_edges('conversation', should_continue, {
        'update_memory': 'update_memory',
        'consult_database': 'consult_database',
        'summarize_conversation': 'summarize_conversation',
    })
workflow.add_edge('consult_database', 'conversation')
workflow.add_edge('summarize_conversation', END)
workflow.add_edge('update_memory', END)

graph = workflow.compile(checkpointer=short_term_memory)


__all__ = ['graph']