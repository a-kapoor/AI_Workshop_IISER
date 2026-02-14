from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'World')
    return jsonify({"message": f"Hello, {name}!"})

@app.route('/api/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify({"received": data})

if __name__ == '__main__':
    app.run(debug=True, port=5000)