import os
import google.generativeai as gogenai
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
from PIL import Image
import io
from google import genai

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno (API Key)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    logger.error("Error: GOOGLE_API_KEY no encontrada en el archivo .env")
    # Podrías lanzar una excepción aquí o manejarlo como prefieras
    # exit()

def calculate_repair_budget(damaged_parts_list, damaged_img):    
    # Aquí puedes implementar la lógica para calcular el presupuesto de reparación
    # Por ahora, devolvemos un valor de ejemplo
    prompt_costo = f"""
    A partir de la siguiente lista de partes de repuesto necesarias para la reparación de un coche tradicional:

    {damaged_parts_list}

    Recuerda que la lista usa la terminología común de repuestos. Considerando los costos promedio de mano de obra y repuestos en Colombia para un Nissan Sentra 2019 (automóvil de combustión interna estándar, no eléctrico), calcula el costo total aproximado de la reparación en pesos colombianos (COP).

    Proporciona una estimación del costo total, incluyendo tanto el valor de los repuestos como la mano de obra asociada a la instalación o reparación de cada parte. Si alguna parte de la lista no es clara o requiere más información para estimar su costo, indícalo brevemente.

    Responde con el costo total estimado en COP y, si es posible, una breve desglose de los costos por categoría (repuestos y mano de obra). Recuerda desglosar todos los costos por cada repuesto necesario y no hablar sobre el modelo del vehiculo.

    Ejemplo de respuesta esperada:
    "Costo total estimado: $1.500.000 COP
    Desglose aproximado:
    - Repuestos: $900.000 COP
    - Mano de obra: $600.000 COP"

    Aqui esta la imagen del vehiculo dañado, para que valides tu respuesta, , si consideras que hace falta alguna parte dañada o sobra alguna en la lista que te entregue, ajustala:
    {damaged_img}
    """
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25", #gemini-2.5-flash-preview-04-17
    contents=prompt_costo,
    )

    return response.text

# Configurar la API de Google Gemini
try:
    gogenai.configure(api_key=API_KEY)
    # Configuración de generación (ajustable)
    generation_config = {
        "temperature": 0, # Controla la aleatoriedad. Más bajo = más determinista.
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096, # Máximo de tokens en la respuesta
    }
    # Configuración de seguridad (ajustable)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = gogenai.GenerativeModel(model_name="gemini-1.5-pro", # O "gemini-1.5-flash" si prefieres
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    logger.info("Modelo Gemini configurado exitosamente.")
except Exception as e:
    logger.error(f"Error configurando el modelo Gemini: {e}")
    # Manejar el error apropiadamente, quizás la app no debería iniciar.
    model = None # Marcar que el modelo no está disponible


# Inicializar FastAPI
app = FastAPI(title="Analizador de Daños de Vehículos")

# Configurar plantillas Jinja2
templates = Jinja2Templates(directory="frontend")

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página HTML principal."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-damage/", response_class=JSONResponse)
async def analyze_vehicle_damage(request: Request, file: UploadFile = File(...)):
    """Recibe una imagen, la analiza con Gemini y devuelve los daños."""
    if not model:
         logger.error("Intento de análisis pero el modelo no está inicializado.")
         return JSONResponse(
            status_code=500,
            content={"error": "El modelo de IA no está disponible. Revisa la configuración y la API Key."}
        )

    logger.info(f"Recibida imagen: {file.filename}, tipo: {file.content_type}")

    # Validar tipo de archivo (opcional pero recomendado)
    if not file.content_type.startswith("image/"):
        logger.warning(f"Tipo de archivo no soportado: {file.content_type}")
        return JSONResponse(
            status_code=400,
            content={"error": "Tipo de archivo no soportado. Por favor sube una imagen (JPEG, PNG, WEBP, etc.)."}
        )

    try:
        # Leer contenido de la imagen
        contents = await file.read()
        logger.info(f"Imagen leída, tamaño: {len(contents)} bytes.")

        # Convertir a objeto PIL Image para Gemini
        img = Image.open(io.BytesIO(contents))

        # Preparar el prompt para Gemini
        # Es CRUCIAL refinar este prompt para obtener los mejores resultados
        prompt_parts = [
            "Analiza detalladamente la siguiente imagen de un vehículo accidentado.",
            "Tu tarea principal es identificar y listar ÚNICAMENTE las partes del carro que están visiblemente dañadas y que probablemente requerirán ser reemplazadas o reparadas mediante un repuesto para solucionar el daño.",
            "Sea concreto y entregue una lista clara de estas partes de repuestos necesarias, separadas por comas.",
            "Intenta identificar las partes del vehículo con la mayor especificidad posible, utilizando la terminología común de repuestos (e.g., 'faro delantero derecho roto', 'parabrisas estrellado', 'guardabarros trasero destrozado'). Recuerda siempre listar la posición de la parte dañada segun el vehiculo y no la imagen",
            "Si no se observan daños evidentes en la imagen, indica claramente: 'No se observan daños visibles en la imagen'.",
            "Ejemplo de respuesta esperada si hay daños: 'Parachoques delantero roto, Faro izquierdo quebrado, Puerta del conductor abollada, Espejo retrovisor derecho desprendido'.",
            "Ejemplo si no hay daños visibles: 'No se observan daños evidentes en la imagen.'\n\n",
            img, # La imagen misma
        ]

        logger.info("Enviando solicitud a Gemini...")
        # Llamar a la API de Gemini
        response = model.generate_content(prompt_parts)
        logger.info("Respuesta recibida de Gemini.")

        # Extraer y limpiar el texto de la respuesta
        damage_report = response.text.strip()
        logger.info(f"Texto del informe de daños: {damage_report}")

        # Podrías hacer un post-procesamiento aquí si quieres una lista más estructurada
        # Por ejemplo, intentar dividir por comas o saltos de línea si Gemini no da una lista perfecta
        damaged_parts_list = [part.strip() for part in damage_report.split(',') if part.strip()]
        if not damaged_parts_list and "no se observan daños" not in damage_report.lower():
             # Si la división por comas falla pero hay texto, usa el texto completo
             damaged_parts_list = [damage_report]
        elif not damaged_parts_list and "no se observan daños" in damage_report.lower():
             damaged_parts_list = ["No se observan daños evidentes."]

        # Crear un presupuesto de reparación basado en las partes dañadas
        repair_budget = calculate_repair_budget(damaged_parts_list, img)

        print(repair_budget)

        return JSONResponse(content={"damaged_parts": damaged_parts_list, "repair_budget": repair_budget})

    except Exception as e:
        logger.error(f"Error durante el análisis de la imagen: {e}", exc_info=True)
        # Considera devolver mensajes de error más específicos si es posible
        return JSONResponse(
            status_code=500,
            content={"error": f"Ocurrió un error procesando la imagen: {e}"}
        )
    finally:
        # Asegurarse de cerrar el archivo si es necesario (FastAPI suele manejarlo bien)
        await file.close()

# --- Ejecución (para desarrollo) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn en http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # El reload=True es útil para desarrollo, se reinicia al guardar cambios.