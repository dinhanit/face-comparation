from main import is_same_person, read_image
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Query
import socket
app = FastAPI()


@app.get("/")
async def root():
    return f"ID: {socket.gethostname()}"

@app.get("/api_recognize")
async def api_recognize(key1: str = Query(...), key2: str = Query(...), key3: str = Query(...)):
    img_origin, img_predict = read_image(key1), read_image(key2)
    result = is_same_person(img_origin, img_predict,thread_hold=float(key3))
    result = bool(result)  # Ensure result is a boolean

    return {"result": result}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

