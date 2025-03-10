# Apartado 2: Preguntas Teóricas

## Diferencias entre 'completion' y 'chat' models

Durante el pre-entrenamiento de un LLM (Large Language Model), el objetivo principal es predecir la siguiente palabra. Desde esta fase, los modelos pueden considerarse como **completion models**.

A partir de esta base, es importante destacar dos aspectos fundamentales que diferencian su capacidad como chatbots:

1. **Fase de Fine-Tuning**: En esta fase, una de las múltiples tareas entrenadas es actuar como asistente o mantener una conversación coherente en cuanto al tono y al vocabulario.
   
2. **Modelo completamente entrenado**: Una vez que el LLM está completamente entrenado (pre-entrenamiento, fine-tuning y RLHF), aunque inicialmente sea muy bueno completando texto, el proporcionar contexto y mensajes previos en cada interacción permite que el modelo funcione efectivamente como un chatbot.

## Forzar respuestas a 'sí' o 'no' y parsear salidas

En lo que yo entiendo, no se puede forzar directamente una respuesta específica (como 'sí' o 'no') en un LLM "out-of-the-box" debido a su naturaleza probabilística. Sin embargo, es posible asegurar que la respuesta del LLM siga un formato predefinido después de generarla. Esto se puede lograr mediante la función **structured_output** (una versión mejorada del **JSON mode**) proporcionada por diferentes proveedores.

Se puede suministrar un esquema a seguir utilizando **clases de Pydantic** y añadir validadores para incrementar la seguridad del formato de salida.

## Ventajas e inconvenientes de RAG vs Fine-Tuning

- **Fine-Tuning**:
  - **Ventaja**: Suele ser más preciso cuando se trata de realizar una nueva tarea especializada. Es ideal para casos donde se desee generar una funcionalidad nueva o de nicho que no haya sido abordada durante el entrenamiento original.
  - **Desventaja**: Uno de los principales inconvenientes es el alto coste en términos de recursos y tiempo.

- **RAG (Retrieval-Augmented Generation)**:
  - **Ventaja**: El acceso a nueva información por parte del LLM es inmediato, con un coste menor y facilidad para actualizarse.
  - **Desventaja**: Añadir capas extra (como **parsing**, **retrieval**, **re-ranking**, etc.) incrementa la complejidad, lo que puede hacer más difícil encontrar un resultado óptimo.

## Evaluación de bots Q&A y RAG

### Evaluación de bots Q&A

Existen varias métricas que se pueden utilizar para evaluar un bot de preguntas y respuestas (Q&A), tales como:

- **Accuracy**
- **Correctness**
- **Hallucination**
- **Prompt Alignment**
- Métricas tradicionales de comparación de textos como **BLEU**, **ROUGE** o **BERTScore**.

Estas métricas pueden ser calculadas usando diferentes sistemas de evaluación, tales como:

- **LLM as a judge**: Se emplea un modelo más potente para evaluar las respuestas del modelo en cuestión.
- **LLM as a jury**: Utiliza varios modelos más pequeños y de diferentes proveedores para evaluar sobre una métrica, actuando como un ensamblador.

Existen librerías como **DeepEval** o **DeepChecks** que ayudan a crear estos sistemas sobre distintas métricas.

### Evaluación de sistemas RAG

En cuanto a los sistemas RAG, las métricas comunes incluyen:

- **Accuracy**
- **Precision**
- **Recall**
- **Relevancy**

También existen sistemas que utilizan otros LLMs para validar distintas partes del sistema RAG, como los **embeddings**, **vector search**, **re-ranks**, **chunk size**, y el valor de **top-k**. Para facilitar la creación de estos sistemas, existen librerías como **RAGAS** o **Giskard**, y herramientas como **DeepEval** también incluyen módulos específicos para la evaluación de sistemas RAG.


# Apartado 4: Explicación entrenamiento modelo de detección de objetos.
## Pasos necesarios para entrenar un modelo con categorías nuevas
- 1º Pensar en el caso de uso para determinar qué preferencias necesitamos **(precisión, latencia de inferencia, costes, infraestructura, etc.)**.

- 2º **Según el caso de uso**, elegimos la arquitectura de modelo más adecuada: por ejemplo, modelos YOLO híbridos que equilibran precisión y latencia, RT-DETR para precisión, YOLO NAS para latencia, etc.

- 3º Recolección de datos y etiquetado. En cuanto a herramientas de etiquetado, a mí siempre me ha gustado utilizar **CVAT**, que es de código abierto, aunque gracias a los desarrollos multimodales de los LLM existen varios servicios para realizar etiquetados asistidos en caso de que las cantidades sean demasiado grandes. Respecto a las cantidades, no sabría contestar de forma exacta, ya que, según mi experiencia, **importa mucho la dificultad de las clases, el entorno donde se ejecuta y la precisión que se requiera**. Aunque, a priori, utilizando arquitecturas bastante potentes como YOLO, no he necesitado más de unas pocas miles de muestras para obtener resultados bastante buenos en tareas de visión por computadora no muy complejas.

- 4º Entrenamiento del modelo. Aquí se podrían utilizar distintas plataformas con acceso a GPUs, aunque si se trata de modelos como YOLO, no se requiere de una infraestructura muy compleja. En el caso de estar sobrepasando los tiempos de entrenamiento se pueden utilizar modelos más simples (tanto de la misma familia como de otras) o se pueden **utilizar versiones quantificadas o cambiar parámetros como learning rate o el batch size**.

- 5º Validación de los resultados. Esta es una parte crucial, donde también hay que tener en cuenta el caso de uso y cuáles son las métricas/clases más importantes. Por ejemplo, puede ser fundamental detectar todas las instancias de una clase de suma importancia, sin importar tanto los falsos positivos (es importante tener un recall alto para esa clase). Además, se deben considerar **métricas específicas de detección de objetos**, como **mAP, IoU, box_loss, clas_loss y dfl_loss**. De nuevo, cuáles son las métricas necesarias dependerán del caso de uso.

- 6º Ajuste de hiperparámetros. Este es un proceso iterativo que se realiza junto con el entrenamiento, donde, analizando las métricas, podemos ajustar tanto los **hiperparámetros del modelo** como de otras partes del pipeline de entrenamiento. Por ejemplo, si observamos overfitting, quizás necesitemos modificar algunos de los hiperparámetros de **regularización** del modelo. Si notamos que el aprendizaje se estanca, puede ser necesario ajustar el **learning rate o aumentar la complejidad del modelo**. Además, si se detecta un desbalanceo de clases o que las imágenes que se verán en producción no son de la calidad adecuada (por ejemplo, con demasiado brillo o contraste, o de mala calidad), se pueden modificar los parámetros de **data augmentation** o utilizar modelos de **super resolución**. Esta parte suele llevar bastante tiempo, ya que existen muchos aspectos que se pueden modificar.

- 7º Desplege del modelo. Creación del endpoint en donde se va a consumir el modelo teniendo en cuenta el **tráfico de uso** al igual que en el entramiento se pueden utilizar tanto versiones quantificadas para la inferencia como **versiones específicas del hardware**. (onnix, openVINO, o las esecificas de librerias como pytorch). Este último punto tambien es importante si el modelo va a correr en el edge.