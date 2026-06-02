from pyngrok import ngrok
import uvicorn
public_url = ngrok.connect(8000)
print(f"Colab Engine URL: {public_url}")
uvicorn.run(app, host="0.0.0.0", port=8000)