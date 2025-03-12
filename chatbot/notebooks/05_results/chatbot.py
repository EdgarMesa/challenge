import os
import sys
import random
import html
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from gradio.themes import Ocean  
from langchain.schema import AIMessage, HumanMessage
from langgraph.types import Command

root_dir = Path(os.getcwd()).parent.parent
sys.path.insert(0, str(root_dir))

load_dotenv(f'{root_dir}/.env')

from src.d00_utils.utils import parse_graph_output
from src.d01_data.graph import graph


# Name of the file to store users
USER_FILE = str(root_dir / 'data' / '03_memories' / 'usuarios.txt')
IS_INTERRUPT = False
    
def load_users():
    """
    Load the list of users from the file.
    If the file does not exist or is empty, default users are created.
    """
    try:
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            users = [line.strip() for line in f if line.strip()]
        if not users:
            users = ['usuario1']
            with open(USER_FILE, 'w', encoding='utf-8') as f:
                for user in users:
                    f.write(user + '\n')
    except FileNotFoundError:
        users = ['usuario1']
        with open(USER_FILE, 'w', encoding='utf-8') as f:
            for user in users:
                f.write(user + '\n')
    return users


def predict(message, history, user_id, debug_mode):
    """
    Convert the history to LangChain format, add the new message, and return
    a simulated response that uses the user ID.
    If debug_mode is "Activado", modify the response accordingly.
    """
    global IS_INTERRUPT
    debug = True if debug_mode == 'Activado' else False
    config = {'configurable': {'thread_id': user_id, 'user_id':user_id}}
    input_message = HumanMessage(content=message)
    
    if not IS_INTERRUPT:
        outputs = graph.invoke({'messages': [input_message]}, config, stream_mode='updates')
    else:
        outputs = graph.invoke(Command(resume=message), config, stream_mode='updates')
        IS_INTERRUPT = False
    node_name = list(outputs[-1].keys())[0]

    if node_name == '__interrupt__':
        IS_INTERRUPT = True
    
    output_messages = parse_graph_output(outputs,
                                         debug=debug)
    if debug:
        formatted_messages = []
        for tag, content in output_messages:
            # Escape special HTML characters
            escaped_content = html.escape(content)
            
            escaped_content = escaped_content.replace("\n", "<br>")

            if tag == 'fn':
                formatted_messages.append(
                    f"<span style='color: orange;'>FUNCTION CALLING MESSAGE<br>"
                    f"--------------------------------</span><br>{escaped_content}"
                )
            elif tag == 'tool':
                formatted_messages.append(
                    f"<span style='color: blue;'>TOOL MESSAGE<br>"
                    f"--------------------------------</span><br>{escaped_content}"
                )
            else:
                formatted_messages.append(
                    f"<span style='color: green;'>AI MODEL MESSAGE<br>"
                    f"--------------------------------</span><br>{escaped_content}"
                )

        return '<br><br>'.join(formatted_messages)
    else:
        return '\n'.join([m for _, m in output_messages])

def add_new_user(user_list, custom_id):
    """
    Adds a new user to the list.
    If a 'custom_id' is provided (and is not empty), it is used as the new ID.
    If the field is empty, a random ID is generated.
    If the custom ID already exists, a random one is generated.
    """
    custom_id = custom_id.strip()
    if custom_id:
        new_id = custom_id
        if new_id in user_list:
            new_id = f'usuario{random.randint(1000, 9999)}'
    else:
        new_id = f'usuario{random.randint(1000, 9999)}'
    user_list.append(new_id)
    with open(USER_FILE, 'a', encoding='utf-8') as f:
        f.write(new_id + '\n')
    return user_list, gr.update(choices=user_list, value=new_id)

# Load users from the file
usuarios_iniciales = load_users()

with gr.Blocks(theme=Ocean()) as demo: 
    gr.Markdown('# Chatbot Leyes Españolas 🇪🇸')
    
    # Hidden state to store the list of user IDs.
    user_list_state = gr.State(usuarios_iniciales)
    
    with gr.Row():
        # Dropdown to select the user ID.
        user_id_dropdown = gr.Dropdown(
            label='Seleccionar ID de Usuario', 
            choices=usuarios_iniciales, 
            value=usuarios_iniciales[0] if usuarios_iniciales else None
        )
        # Radio button to activate/deactivate debug mode.
        debug_radio = gr.Radio(
            label='Modo Depuración',
            choices=['Activado', 'Desactivado'],
            value='Desactivado'
        )
    
    with gr.Row():
        # Textbox for the new user ID (optional) with a button to add it.
        new_user_input = gr.Textbox(
            label='Nuevo ID de Usuario (opcional)', 
            placeholder="Ingresa un ID o deja en blanco para uno aleatorio",
            lines=1
        )
        add_user_button = gr.Button('Agregar Usuario')
    
    # The ChatInterface receives additional inputs: the user ID and the debug mode.
    chat_interface = gr.ChatInterface(
        fn=predict,
        type='messages',
        additional_inputs=[user_id_dropdown, debug_radio],
        title='Chatbot',
        description='Interfaz de chat con selección de usuario y características de depuración.'
    )
    
    # When clicking the 'Agregar Usuario' button, a new ID is generated (using the input if provided),
    # the state and dropdown are updated, and it is written to the file.
    add_user_button.click(
        add_new_user,
        inputs=[user_list_state, new_user_input],
        outputs=[user_list_state, user_id_dropdown]
    )
    

demo.launch()