from flask import Flask, request, jsonify, render_template
from fill_level_calculator import run_tank_network_analysis   # NEW
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable cross-origin for frontend JS

@app.route('/')
def home():
    return render_template('test_fill_optmizn.html')  # Serves new.html from /templates

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json  # expects {"tanks": [...], "adjacency": [...], "environment": {...}}
        tanks = data.get('tanks', [])
        adjacency = data.get('adjacency', [])
        environment = data.get('environment', {})

        # Run your network calculator
        result = run_tank_network_analysis(tanks, adjacency, environment)

        # result will already contain tanks, adjacency, plot, etc.
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
