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

Prompts: 

Shocking secret government exposed hiding alien technology for decades
Scientists discover miracle cure for cancer Big Pharma doesn't want you to know
Breaking: Obama born in Kenya, new documents exposed
Warning: 5G towers spreading dangerous virus, urgent truth revealed


Federal Reserve raises interest rates by 0.25 percent
Apple reports record quarterly earnings amid strong iPhone sales
NASA announces new Mars mission scheduled for 2028
Senate passes bipartisan infrastructure bill after months of negotiations


You won't believe what happened at the White House today
Doctors reveal one weird trick to lose weight fast
Celebrity spotted leaving hospital, condition unknown