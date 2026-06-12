Activate the venv first - source/venv/bin/activate(Linux+Mac)
.\venv\Scripts\activate.bat(Windows+PowerShell)

Trains the model(no need to run this if fakenews_model.pkl is present)
python app.py

Starts the server 
python server.py

In a new terminal(CORS issues)
python -m http.server 8080

Go to 
http://localhost:8080/index.html