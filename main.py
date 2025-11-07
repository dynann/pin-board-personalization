from fastapi import FastAPI, UploadFile, File, HTTPException
from service import vector_service
from fastapi.middleware.cors import CORSMiddleware
import logging
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello():
    return { "message": "hello 404 not found" }

@app.post("/vectorize-pins")
async def vectorize_pin(file: UploadFile = File(...)):
    print("hello world")
    logging.info(f"vectorizing pin name: {file.filename}")
    
    #validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_data = await file.read()

    #get image vector
    image_vector = vector_service.vector_service.image_to_vector(image_data)
    logging.info(f"successfully vectorized pin, vector length {len(image_vector)}")

    return {
        "status": "success",
        "vector": image_vector,
        "vector_dimension": len(image_vector),
        "model": "openai-clip-vit-base-patch32",
        "filename": file.filename
    }