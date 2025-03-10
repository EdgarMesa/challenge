from pydantic import BaseModel, Field

class Resumen(BaseModel):
    resumen: str = Field(title='Resumen', description='Resumen de la conversación')