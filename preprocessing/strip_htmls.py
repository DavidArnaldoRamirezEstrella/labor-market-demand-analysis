import os
import sys
import unicodedata

CRAWLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS_DIR = os.path.join(CRAWLER_DIR,"block_tags")

body_tags_txt    = os.path.join(TAGS_DIR,"body_tags.txt")
benefit_tags_txt = os.path.join(TAGS_DIR,"benefit_tags.txt")
req_tags_txt     = os.path.join(TAGS_DIR,"req_tags.txt")
func_tags_txt    = os.path.join(TAGS_DIR,"func_tags.txt")

text_data = ["\n            \u00daNETE AL MEJOR EQUIPO!", "\r\n", "\r\nAtento, empresa transnacional l\u00edder en servicios de Contact Center y el \u00fanico dentro de las mejores empresas para trabajar seg\u00fan el ranking Great Place To Work.", "\r\n", "\r\nSe encuentra en la b\u00fasqueda de j\u00f3venes que posean un alto nivel comunicaci\u00f3n, capacidad de escucha y vocaci\u00f3n para incursionar en el \u00e1rea de Atenci\u00f3n al cliente.", "\r\n", "\r\n> FUNCIONES:", "\r\n", "\r\nRecepcionar llamadas de clientes para aclarar dudas relacionadas a su telefon\u00eda m\u00f3vil.", "\r\n", "\r\n> REQUISITOS:", "\r\n", "\r\n- Edad: 18 a m\u00e1s", "\r\n- Estudios: T\u00e9cnicos o Universitarios (truncos o en curso)", "\r\n- Condici\u00f3n: Full time", "\r\n- Experiencia: Deseable M\u00ednimo 3 meses en el \u00e1rea de atenci\u00f3n al cliente.", "\r\n- Disponibilidad: En horarios Full Time \u2013 Part time por las TARDES.", "\r\n- Residencia: Vivir en zonas aleda\u00f1as al CALLAO.", "\r\n- Conocimiento y manejo de office a nivel usuario.", "\r\n", "\r\n> BENEFICIOS:", "\r\n", "\r\n- Planilla con todos los beneficios de ley.", "\r\n- Movilidad de retorno a casa para los turnos que terminen a partir de las 23:00 horas.", "\r\n- Capacitaci\u00f3n constante.", "\r\n- L\u00ednea de Carrera", "\r\n- Descuentos Corporativos en entidades educativas y recreacionales.", "\r\n- Excelente Clima Laboral.", "\r\n- Bonos por desempe\u00f1o.", "\r\n", "\r\n", "\r\n", "* Ac\u00e9rcate \u00e9ste lunes 10 de febrero en los siguientes horarios de: 8:30 am \u00f3 3:00 pm con tu DNI., en Av. Maquinarias 6015 Carmen de La Legua (cruce de Av. Argentina y Jr. Pac\u00edfico a 3 cuadras de Av. Universitaria) Callao.", "\r\n", "\r\n*Preguntar por: Luisa S\u00e0nchez.        "]

#text_data = ["\n            Nos encontramos en b\u00fasqueda de personas din\u00e1micas, con deseos de superaci\u00f3n y de hacer l\u00ednea de carrera con nosotros, con o sin experiencia en ventas de preferencia personas Mayores, edad promedio de 18 a 45 a\u00f1os, trato directo con clientes v\u00eda call center, con disponibilidad de trabajar en turnos ma\u00f1ana y tarde.", "\r\n", "\r\nREQUISITOS:", "\r\n", "\r\n-Edad: Mayores de 18 a\u00f1os.", "\r\n- C/S Experiencia en ventas", "\r\n- Estudios t\u00e9cnicos o universitarios (truncos, en curso o culminados)", "\r\n- Trabajo bajo presi\u00f3n y responsable.", "\r\n- Manejo de office a nivel usuario.", "\r\n- Din\u00e1micos, con buen nivel de comunicaci\u00f3n, tolerancia, iniciativa, responsabilidad, capacidad de persuasi\u00f3n, empat\u00eda y habilidades sociales.", "\r\n- Disponibilidad para laborar en horario de MA\u00d1ANA Y TARDE.", "\r\n", "\r\n\u2022\t7:30am a 4:00pm de lunes a s\u00e1bado", "\r\n\u2022\t8:00am a 5:00pm de lunes a s\u00e1bado", "\r\n\u2022\t9:00am a 6:00pm de lunes a viernes y s\u00e1bados de 9:00am a 1:00pm ", "\r\n\u2022\t3:00pm a 10:00pm de lunes a viernes y s\u00e1bados de 9:00am a 5:00pm ", "\r\n\u2022\t4:00pm a 10:30pm de lunes a s\u00e1bado ", "\r\n\u2022\tTodos los horarios son fijos ", "\r\n", "\r\nOFRECEMOS:", "\r\n", "\r\n-Sueldo por encima del mercado, incentivos diarios y mensuales (pagos quincenales).", "\r\n-Capacitaciones remuneradas.", "\r\n-Grato clima laboral.", "\r\n-L\u00ednea de Carrera.", "\r\n-Puntualidad en los pagos.", "\r\n-Oportunidad de desarrollar una l\u00ednea de carrera.", "\r\n-Formaci\u00f3n constante para el desarrollo en la gesti\u00f3n y cumplimiento de las metas.", "\r\n-Convenios con instituciones educativas y otras empresas para tu beneficio y el de tu familia.", "\r\n", "\r\nGimnasio Libre:", "\r\n-Clases de aer\u00f3bicos, Tae Bo, Bailes, Pilates, etc. con un personal trainner.", "\r\n", "\r\nSala de Masajes:", "\r\n-Contamos con una masajista profesional y los ambientes adecuados.", "\r\nAc\u00e9rcate: Con tu CV (Copia y original) a nuestra Sede: Av. Arequipa 2618 piso 7 - San Isidro (Ref. Cruce de la Av. Javier Prado y la Av. Arequipa a espaldas de HYUNDAI de Javier Prado Oeste Cdra. 1), de L-V de 8 am a 1 pm y de 2 pm a 5 pm. RPC: 989142248.", ".        "]

def especial(car):
    test = [car==spe for spe in ['\n','\r','\t','\a','\b',' ']]
    cont = 0
    for t in test:
        if t == True:
            cont+=1
    
    if cont:
        return True
    else:
        return False    
    
def strip_encode(text):
    ans = []
    text = [unicodedata.normalize('NFKD', line).encode('ascii', 'ignore') for line in text]
    for texto in text:
        texto = texto.lower()
        while(len(texto)>0 and (especial(texto[-1]) or especial(texto[0]) )):
            texto = texto.strip(' ')
            texto = texto.strip('\n')
            texto = texto.strip('\t')
            texto = texto.strip('\r')
        
        if len(texto)>1:
            ans.append(texto)
        
    return ans

def leer_tags(data):
    ans = []
    line = data.readline()
    while line:
        ans.append( line )
        line = data.readline()
    return ans

text_data = strip_encode(text_data)
"""
func_tags    = strip_encode(leer_tags(open(func_tags_txt)))
req_tags     = strip_encode(leer_tags(open(req_tags_txt)))
body_tags    = strip_encode(leer_tags(open(body_tags_txt)))
benefit_tags = strip_encode(leer_tags(open(benefit_tags_txt)))

tags = [func_tags, req_tags, body_tags, benefit_tags]

def blocks(text):
    ans = []
        
    for i in range(len(text)):
        line = text[i]
        for j in range(len(tags)):
            encontro = False
            for k in range(len(tags[j])):
                tag = tags[j][k]
                if line.find(tag)!=-1 and len(line)-len(tag)<5:
                    ans.append((j,i))
                    encontro = True
                    break
            if encontro:
                break
    return ans
    
"""
"""                    
print func_data
print req_data
print body_data
print benefit_data
print text_data
"""

print text_data
#bloques = blocks(text_data)
#print bloques



