from flask import Flask, render_template_string, request, jsonify
import threading

app = Flask(__name__)

robot_status = {}
status_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Monitorizare Robot</title>
    <meta http-equiv="refresh" content="1"> <!-- Refresh la fiecare secundă -->
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        pre { background: #eee; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dashboard monitorizare robot</h1>
        <h3>Stare curentă:</h3>
        <pre id="status-data">Așteptare date...</pre>
    </div>
    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/get_status');
                const data = await response.json();
                document.getElementById('status-data').textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                document.getElementById('status-data').textContent = 'Eroare la conectare...';
            }
        }
        setInterval(fetchStatus, 500); // Actualizează la 500ms
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/robot/status', methods=['POST'])
def receive_status():
    global robot_status
    data = request.json
    
    with status_lock:
        robot_status = data
        
    return jsonify({"status": "ok"}), 200

@app.route('/get_status', methods=['GET'])
def get_status():
    with status_lock:
        return jsonify(robot_status)

if __name__ == '__main__':
    print("Serverul de monitorizare: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)