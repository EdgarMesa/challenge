from pydantic import BaseModel, Field
from typing import Literal

class Resumen(BaseModel):
    resumen: str = Field(title='Resumen', description='Resumen de la conversación.')
    
class PerfilUsuario(BaseModel):
    nombre: str = Field(title='Nombre', description='Nombre del usuario. Si no existe, dejarlo vacio.')
    profesion: str = Field(title='Profesion', description='Profesion o trabajo del usuario. Si no existe, dejarlo vacio.')
    gmail: str = Field(title='Gmail', description='Gmail o Email del usuario. Si no existe, dejarlo vacio.')
    actualizar: Literal['si', 'no'] = Field(title='Actualizacion', description="""Si la información extraida es nueva de la que ya esta en la base de datos.
En el caso de que si exista nueva habrá que responder 'si' en el caso de que la información sea la misma o no tiene responder 'no'.""")