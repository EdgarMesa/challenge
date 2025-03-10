import re

def extract_index_from_doc(doc):
    """
    Extracts the index pages from a document and returns the remaining document after the index.

    Parameters:
    doc (list): A list of page objects, each with a 'get_text' method and a 'number' attribute.

    Returns:
    tuple: A tuple containing:
        - The remaining document pages after the index.
        - A list of pages that are part of the index.
    """
    indexes_pages = []  

    for page in doc:
        text = page.get_text()
        
        # If the page contains the marker 'TEXTO CONSOLIDADO', it signals the end of the index section.
        if 'TEXTO CONSOLIDADO' in text:
            break
        
        # If the marker is not found, add the page to the index pages list.
        indexes_pages.append(page)

    index_last_page = max([p.number for p in indexes_pages])
    
    # Update the document by slicing it to exclude the index pages.
    # The remaining document starts after the last index page.
    doc = doc[index_last_page + 1:]

    print('Última página del índice', index_last_page + 1)
    
    return doc, indexes_pages


def get_hierarchy(text):
    """
    Extracts hierarchical elements from the given text based on a regular expression pattern.
    
    Parameters:
    text (str): The input text to be processed for hierarchical elements.
    
    Returns:
    list: A list of strings representing the hierarchy levels extracted and formatted.
    """
    jerarquias_pattern = r"^(Parte dispositiva|Preámbulo|LIBRO|TÍTULO|CAPÍTULO|Sección).*$"
    jerarquia = []

    for line in text.splitlines():
        line = line.split('.')[0]
        
        # Check if the line matches the pattern defined by the variable 'jerarquias'
        if re.match(jerarquias_pattern, line, re.IGNORECASE):
            jerarquia.append(line.strip())

    if 'Preámbulo' in jerarquia:
        jerarquia[0] = jerarquia[0].upper()
        
    for i, jer in enumerate(jerarquia):
        if 'bis' in jer.lower():
            jerarquia[i] = jerarquia[i].upper()
    
    return jerarquia

def get_hierarchy_level(item):
    """
    Determina el nivel jerárquico de un elemento basado en su prefijo textual.
    
      - Si el elemento comienza con "sección", se retorna el nivel 4.
      - Si el elemento comienza con "capítulo", se retorna el nivel 3.
      - Si el elemento comienza con "título", se retorna el nivel 2.
      - Si el elemento comienza con "libro", "preámbulo" o "parte dispositiva", se retorna el nivel 1.
      - En caso contrario, se retorna 0.
      
    Parámetros:
    item (str): El texto que representa el ítem a evaluar.
    
    Retorna:
    int: El nivel jerárquico asignado al ítem.
    """
    item = item.lower()
    
    if item.startswith('sección'):
        return 4
    elif item.startswith('capítulo'):
        return 3
    elif item.startswith('título'):
        return 2
    elif item.startswith('libro') or item.startswith('preámbulo') or item.startswith('parte dispositiva'):
        return 1
    else:
        return 0
    
def extract_previous_hierarchy(texto, articles, jerarquia):
    """
    Extracts the hierarchy preceding each article header found in the text.
    
    Parameters:
    texto (str): The full text from which to extract hierarchies.
    jerarquia (list): A list of hierarchy header strings to be used in the extraction.
    
    Returns:
    dict: A dictionary where keys are article headers (with a trailing period) and values are lists
          of hierarchy items found before that article header.
    """
    
    # Build a regex pattern to match any of the hierarchy headers at the beginning of a line.
    extract_jerar = r'(?m)^(' + '|'.join(map(re.escape, jerarquia)) + r')\b'
    
    final_hierarchy = {}

    for match in articles:
        match = match.split('.')[0]
        
        # Create a regex pattern to match the full header of the article.
        # This pattern articles lines starting with the 'match' followed by a period.
        pattern = rf'^{match}\.'
        
        article_match = re.search(pattern, texto, flags=re.MULTILINE)

        if article_match:
            # If the article header is found, take all text that comes before it.
            before_text = texto[:article_match.start()]
            
            # Find all occurrences of hierarchy elements in the 'before_text' using the regex pattern.
            prev_jerar = re.findall(extract_jerar, before_text)
            
            final_hierarchy[f'{match}.'] = prev_jerar
            
        else:
            print(f'Article {match} not found.')
            final_hierarchy[f'{match}.'] = []

    return final_hierarchy


def process_sections(text):
    """
    Process the input string to extract sections for LIBRO, TÍTULO, CAPÍTULO, and SECCIÓN.
    Args:
        text (str): The input string containing the sections.
    
    Returns:
        str: A formatted string with each section in the format:
             libro:<value>
             titulo:<value>
             capitulo:<value>
             seccion:<value>
    """
    libro_match = re.search(r'(?i)LIBRO\s+([^\n]+)', text)
    titulo_match = re.search(r'(?i)T[IÍ]TULO\s+([^\n]+)', text)
    capitulo_match = re.search(r'(?i)CAP[IÍ]TULO\s+([^\n]+)', text)
    seccion_match = re.search(r'(?i)SECCI[ÓO]N\s+([^\n]+)', text)
    
    # If a section is found, construct the output using the found text.
    # Otherwise, default to 'SIN <SECTION>'.
    libro    = f'LIBRO {libro_match.group(1).strip()}'    if libro_match else 'SIN LIBRO'
    titulo   = f'TÍTULO {titulo_match.group(1).strip()}'   if titulo_match else 'SIN TÍTULO'
    capitulo = f'CAPÍTULO {capitulo_match.group(1).strip()}' if capitulo_match else 'SIN CAPÍTULO'
    seccion  = f'SECCIÓN {seccion_match.group(1).strip()}'   if seccion_match else 'SIN SECCION'
    
    return f'libro:{libro}\ntitulo:{titulo}\ncapitulo:{capitulo}\nseccion:{seccion}'


def extract_final_hierarchy(hierarchy, origen='ce'):
    """
    Extracts the final hierarchy (parent elements) for each article, based on hierarchical levels.

    Parameters:
    origen (str): A string that defines the origin. If it is 'ce', the maximum level is set to 2; otherwise, to 1.

    Returns:
    dict: A dictionary where each key is an article and each value is a list of the extracted hierarchical parent elements.
    """
    # Define the maximum level based on the origin: 2 if the origin is 'ce', otherwise 1.
    max_level = 2 if origen == 'ce' else 1
    final_parents = {}

    # It is assumed that 'hierarchy' is a global dictionary containing the hierarchy for each article.
    for art, jerar in hierarchy.items():
        # Reverse the hierarchy list to process it from the end.
        jerar = jerar[::-1]
        # Get the hierarchical levels for each element in the reversed list.
        levels = [get_hierarchy_level(j) for j in jerar]

        parents = []
        parents_level = []

        # Iterate over the elements and their corresponding levels with an index.
        for i, (jerar_prev, nivel_prev) in enumerate(zip(jerar, levels)):
            if i == 0:
                # The first element is directly taken as a parent.
                parents.append(jerar_prev)
                parents_level.append(nivel_prev)
                
            # For the following elements, add them if their level is not already in the list and is lower than the last added level.
            elif nivel_prev not in parents_level and nivel_prev < parents_level[-1]:
                parents.append(jerar_prev)
                parents_level.append(nivel_prev)
            
            # Stop the iteration if the current level matches the maximum allowed level.
            if nivel_prev == max_level:
                break

        final_parents[art] = process_sections('\n'.join(parents))

    return final_parents


def extract_paragraphs(text, search_words):
    """
    Given a text and a list of marker words, this function finds the first occurrence
    of each marker in the text and extracts the corresponding paragraph.
    Parameters:
      text (str): The text to search through.
      search_words (list of str): The list of marker words to look for.
      
    Returns:
      dict: A dictionary where each key is a marker word and the value is the extracted paragraph.
    """
    # For each marker, find its first occurrence in the text.
    occurrences = []
    for word in search_words:
        idx = text.find(word)
        if idx != -1:
            occurrences.append((idx, word))
    
    # Sort the occurrences by their index in the text.
    occurrences.sort(key=lambda x: x[0])
    
    result = {}
    # For each marker, extract from its index until the next marker's index (or the end).
    for i, (start_index, word) in enumerate(occurrences):
        if i < len(occurrences) - 1:
            end_index = occurrences[i + 1][0]
        else:
            end_index = len(text)
        paragraph = text[start_index:end_index].strip()
        result[word] = paragraph
    return result



