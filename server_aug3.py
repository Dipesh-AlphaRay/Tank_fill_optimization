from flask import Flask, request, jsonify
from fill_level_calculator import compute_MinFill_cone
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable cross-origin for frontend JS

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json  # expects {"tanks": [{D, h_f, G}], "V": 120}
        tanks = data.get('tanks', [])
        env = data.get("environment", {})

        # Dynamically extract V and h_f
        V = env.get("Wind Speed (m/s)", 120)  # fallback to 120 if not provided
        h_f = env.get("Flood Height (m)", 10) # fallback to 10 if not provided

        results = []
        for t in tanks:
            res = compute_MinFill_cone(V=V, D=t["diameter"], h_f=h_f, G=t["density"])
            res["Tank ID"] = t.get("id", "N/A")
            results.append(res)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
