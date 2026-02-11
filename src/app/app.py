from flask import Flask, render_template_string, request, jsonify
import threading

app = Flask(__name__)

robot_status = {}  # Variabilă globală pentru stocarea ultimei stări a robotului
status_lock = threading.Lock()  # Lock pentru a preveni Race Conditions între thread-ul serverului și request-urile concurente

# Template HTML/JS integrat direct (pentru simplitate într-un singur fișier)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Monitorizare Robot</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #eef2f7; }
        .container { max-width: 800px; margin: auto; background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .data-section { display: flex; gap: 20px; flex-wrap: wrap; }
        .status-box { flex: 1; min-width: 300px; }
        .chart-box { flex: 1; min-width: 300px; height: 300px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; font-size: 14px; overflow-x: auto; }
        .label { font-weight: bold; color: #34495e; margin-bottom: 10px; display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dashboard Monitorizare Robot AGV</h1>
        <div class="data-section">
            <div class="status-box">
                <span class="label">Stare curentă (JSON):</span>
                <pre id="status-data">Așteptare date...</pre>
            </div>
            <div class="chart-box">
                <span class="label">Probabilitati clase:</span>
                <canvas id="probChart"></canvas>
            </div>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('probChart').getContext('2d');
        const probChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Stânga', 'Înainte', 'Dreapta'],
                datasets: [{
                    label: 'Probabilitate (%)',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(75, 192, 192, 0.7)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Procent (%)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        async function fetchStatus() {
            try {
                const response = await fetch('/get_status');
                const data = await response.json();
                document.getElementById('status-data').textContent = JSON.stringify(data, null, 2);
                
                if (data.probabilities) {
                    probChart.data.datasets[0].data = data.probabilities;
                    probChart.update('none'); // Update without animation for smoother real-time feel
                }
            } catch (e) {
                document.getElementById('status-data').textContent = 'Eroare la conectare...';
            }
        }
        setInterval(fetchStatus, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Randează pagina principală a dashboard-ului."""
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